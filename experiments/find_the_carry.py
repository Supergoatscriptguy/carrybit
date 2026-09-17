"""Where does the carry come from? Three measurements on a trained reversed, zero-padded,
position-coupled addition model, at the moment it predicts answer digit i:

1. Attention mass by source column: the same column (a_i, b_i), the column below
   (a_{i-1}, b_{i-1}), its own previous output digit c_{i-1}, or anything else.
2. Teacher-forced accuracy on digit i as a function of how long the carry chain feeding
   it is (a run of columns summing to exactly 9 below a column that generates a carry).
3. A counterfactual: corrupt c_{i-1} in the prefix so that it implies the opposite carry
   and count how often the prediction for c_i follows it.

    uv run python experiments/find_the_carry.py [run_dir]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from carrybit.arithmetic import Arithmetic, add_digits
from carrybit.config import load_config
from carrybit.model import Transformer
from carrybit.plotting import save, use_style

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/addition_no_wd/position_coupling_s0")
device = "cuda"
N = 20
BATCH = 512

raw = json.loads((run_dir / "config.json").read_text())
task_overrides = [f"task.{k}={json.dumps(v)}" for k, v in raw["task"].items() if k not in ("kind", "test_digits")]
cfg = load_config("configs/addition.yaml", task_overrides + ["task.test_digits=[5]", "task.test_examples=8"])
task = Arithmetic(cfg.task, 123, device)
model = Transformer(16, cfg.model).to(device).eval()
last = max(run_dir.glob("step_*.pt"), key=lambda f: int(f.stem[5:]))
model.load_state_dict(torch.load(last, map_location=device))
gen = torch.Generator(device=device).manual_seed(0)


def problems(a, b):
    n = torch.full((len(a),), N, device=device)
    return task.build(a, b, n, n, train=False)


def random_digits(batch):
    d = torch.randint(0, 10, (batch, N), device=device, generator=gen)
    d[:, -1] = torch.randint(1, 10, (batch,), device=device, generator=gen)
    return d


@torch.no_grad()
def attention_by_source():
    a, b = random_digits(BATCH), random_digits(BATCH)
    ex = problems(a, b)
    for block in model.blocks:
        block.attn.record = True
    model(ex["tokens"], ex["positions"])
    p = ex["prompt_len"]
    # Token layout: $ a_0..a_{N-1} + b_0..b_{N-1} = c_0..c_N $
    a_pos = lambda i: 1 + i
    b_pos = lambda i: 2 + N + i
    c_pos = lambda i: p + i
    categories = ["same column", "column below", "previous output", "delimiters", "other digits"]
    delimiters = [0, 1 + N, 2 + 2 * N]
    mass = np.zeros((len(model.blocks), cfg.model.n_heads, len(categories)))
    # Predicting c_i happens at the token before it, which is c_{i-1} (or = for i=0).
    queries = range(2, N + 1)
    for layer, block in enumerate(model.blocks):
        pat = block.attn.pattern.float().mean(0)  # heads x T x T
        block.attn.record = False
        for i in queries:
            q = c_pos(i - 1)
            row = pat[:, q]
            same = row[:, [a_pos(i), b_pos(i)]].sum(-1)
            below = row[:, [a_pos(i - 1), b_pos(i - 1)]].sum(-1)
            prev = row[:, c_pos(i - 1)]
            delim = row[:, delimiters].sum(-1)
            other = 1 - same - below - prev - delim
            mass[layer] += torch.stack([same, below, prev, delim, other], -1).cpu().numpy()
    return categories, mass / len(queries)


def chain_problems(batch, length, top):
    """Column `top` receives a carry that originated `length` columns below and rippled
    through columns summing to exactly 9. Other columns sum to at most 8 so no other
    carry reaches the chain."""
    a = torch.randint(0, 5, (batch, N), device=device, generator=gen)
    b = torch.randint(0, 4, (batch, N), device=device, generator=gen)
    origin = top - length - 1
    for j in range(origin + 1, top):
        a[:, j] = torch.randint(0, 10, (batch,), device=device, generator=gen)
        b[:, j] = 9 - a[:, j]
    a[:, origin] = torch.randint(1, 10, (batch,), device=device, generator=gen)
    b[:, origin] = torch.maximum(torch.randint(0, 10, (batch,), device=device, generator=gen), 10 - a[:, origin])
    a[:, -1] = a[:, -1].clamp(min=1)
    b[:, -1] = b[:, -1].clamp(min=1)
    return a, b


@torch.no_grad()
def teacher_forced_digit(tokens, positions, pos):
    logits = model(tokens, positions)
    return logits[:, pos - 1].argmax(-1)


@torch.no_grad()
def carry_chain_curve(top=16, max_len=14):
    accs, flips = [], []
    for length in range(0, max_len + 1):
        a, b = chain_problems(BATCH, length, top)
        total, carries = add_digits(a, b)
        assert (carries[:, top] == 1).all()
        ex = problems(a, b)
        p = ex["prompt_len"]
        pred = teacher_forced_digit(ex["tokens"], ex["positions"], p + top)
        accs.append((pred == total[:, top]).float().mean().item())
        # Corrupt c_{top-1} to the digit it would be without its incoming carry. Only
        # meaningful when that column sums to 9, so that both digits are consistent inputs.
        if length == 0:
            flips.append(float("nan"))
            continue
        corrupted = ex["tokens"].clone()
        corrupted[:, p + top - 1] = (total[:, top - 1] - 1) % 10
        pred2 = teacher_forced_digit(corrupted, ex["positions"], p + top)
        flips.append((pred2 == (total[:, top] - 1) % 10).float().mean().item())
    return accs, flips


if __name__ == "__main__":
    categories, mass = attention_by_source()
    accs, flips = carry_chain_curve()
    for layer in range(len(mass)):
        print(f"layer {layer}: " + "  ".join(f"{c} {mass[layer, :, k].mean():.2f}" for k, c in enumerate(categories)))
    print("chain length, accuracy, follows corrupted previous digit:")
    for length, (acc, flip) in enumerate(zip(accs, flips)):
        print(f"  {length:2d}  {acc:.3f}  {flip:.3f}")

    use_style()
    fig, (ax_attn, ax_chain) = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    n_layers, n_heads, _ = mass.shape
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(n_layers * n_heads)
    bottom = np.zeros(len(x))
    for k, c in enumerate(categories):
        vals = mass[:, :, k].reshape(-1)
        ax_attn.bar(x, vals, bottom=bottom, color=colors[k] if k < 3 else ("0.75", "0.9")[k - 3], label=c, width=0.8)
        bottom += vals
    ax_attn.set(xticks=x, xticklabels=[f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)],
                ylabel="attention mass", ylim=(0, 1))
    ax_attn.tick_params(axis="x", labelsize=7, rotation=90)
    ax_attn.set_title("attention while predicting an answer digit", loc="left")
    ax_attn.legend(fontsize=7, loc="upper right", ncol=2)

    ax_chain.plot(range(len(accs)), accs, marker="o", label="digit correct")
    ax_chain.plot(range(len(flips)), flips, marker="o", label="follows corrupted c$_{i-1}$")
    ax_chain.set(xlabel="carry chain length feeding the digit", ylabel="fraction", ylim=(-0.03, 1.03))
    ax_chain.set_title("carry chain stress test", loc="left")
    ax_chain.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, "find_the_carry")
