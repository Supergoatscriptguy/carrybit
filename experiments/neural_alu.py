"""The original joke, measured. Run many copies of a trained addition model on the GPU as
"virtual cores", count additions per second and accuracy, and compare with the GPU doing
the same additions in hardware.

    uv run python experiments/neural_alu.py [run_dir]

Run this with nothing else on the GPU.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from carrybit.arithmetic import Arithmetic
from carrybit.config import load_config
from carrybit.model import Transformer
from carrybit.plotting import save, use_style
from carrybit.tokenizer import PAD

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/addition_no_wd/position_coupling_s0")
device = "cuda"
DIGITS = (5, 10, 20, 40)
BATCH = 2048

raw = json.loads((run_dir / "config.json").read_text())
overrides = [f"task.{k}={json.dumps(v)}" for k, v in raw["task"].items() if k not in ("kind", "test_digits")]
cfg = load_config("configs/addition.yaml", overrides + [f"task.test_digits={list(DIGITS)}", f"task.test_examples={BATCH}"])
task = Arithmetic(cfg.task, 7, device)
model = Transformer(16, cfg.model).to(device).eval()
model.load_state_dict(torch.load(max(run_dir.glob("step_*.pt"), key=lambda f: int(f.stem[5:])), map_location=device))


def timed(fn, repeats=3):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeats


@torch.no_grad()
def model_rate(n: int):
    ex = task.test_sets[n]
    p = ex["prompt_len"]
    n_new = ex["tokens"].shape[1] - p
    with torch.autocast("cuda", dtype=torch.bfloat16):
        seconds = timed(lambda: model.generate(ex["tokens"][:, :p], n_new, ex["positions"]))
        out = model.generate(ex["tokens"][:, :p], n_new, ex["positions"])[:, p:]
    expected = ex["answer"][:, p:]
    acc = ((out == expected) | (expected == PAD)).all(1).float().mean().item()
    # Every generated token is a full forward pass over the prefix, so the cost is quadratic
    # in the answer length. Two flops per weight per token, attention ignored.
    tokens_processed = sum(p + i for i in range(n_new))
    flops = 2 * model.n_params() * tokens_processed
    return BATCH / seconds, acc, flops


def hardware_rate():
    x = torch.randint(0, 2**62, (1 << 24,), device=device)
    y = torch.randint(0, 2**62, (1 << 24,), device=device)
    return x.numel() / timed(lambda: torch.add(x, y), repeats=20)


if __name__ == "__main__":
    hw = hardware_rate()
    rows = [(n, *model_rate(n)) for n in DIGITS]
    print(f"torch.add on the GPU: {hw:.3g} additions/s")
    print("digits  model adds/s   exact match   flops per add   hardware adds per model add")
    for n, rate, acc, flops in rows:
        print(f"{n:6d}  {rate:12.1f}  {acc:12.3f}  {flops:14.3g}  {hw / rate:12.3g}")

    use_style()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    labels = ["torch.add"] + [f"model, {n} digits" for n in DIGITS]
    rates = [hw] + [r[1] for r in rows]
    colors = ["0.5"] + [plt.rcParams["axes.prop_cycle"].by_key()["color"][0]] * len(DIGITS)
    bars = ax.bar(labels, rates, color=colors, width=0.7)
    for bar, (n, rate, acc, flops) in zip(bars[1:], rows):
        ax.text(bar.get_x() + bar.get_width() / 2, rate * 1.4, f"{acc:.0%} correct", ha="center", fontsize=8, color="0.3")
    ax.set(yscale="log", ylabel="additions per second", ylim=(1, hw * 30))
    ax.set_title(f"one RTX 5070 Ti, batch of {BATCH} problems", loc="left")
    fig.tight_layout()
    save(fig, "neural_alu")
