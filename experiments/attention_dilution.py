"""Is length generalization failure attention dilution? Three measurements on position
coupling models:

1. For each checkpoint of a run, the attention mass the digit-adder head puts on the two
   digits of the current column, as a function of test length, next to exact match.
2. That sharpness at a fixed length across training, for runs with and without weight
   decay, next to their out-of-distribution accuracy.
3. Exact match against test length when the attention logits are scaled up at inference.

    uv run python experiments/attention_dilution.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from carrybit.arithmetic import Arithmetic
from carrybit.config import load_config
from carrybit.model import Transformer
from carrybit.plotting import read_metrics, save, use_style
from carrybit.tokenizer import PAD

device = "cuda"
torch.cuda.set_per_process_memory_fraction(0.85)  # see train.py
RUNS = {"weight decay 0.1": "runs/addition/position_coupling_s{}", "no weight decay": "runs/addition_no_wd/position_coupling_s{}"}
SEEDS = (0, 1, 2)
LENGTHS = (10, 20, 30, 40, 50, 60)
SCALES = (1.0, 1.25, 1.5, 2.0)
EXAMPLES = 256


def load(run_dir: Path):
    raw = json.loads((run_dir / "config.json").read_text())
    overrides = [f"task.{k}={json.dumps(v)}" for k, v in raw["task"].items() if k not in ("kind", "test_digits")]
    overrides += [f"model.{k}={json.dumps(v)}" for k, v in raw["model"].items()]
    cfg = load_config("configs/addition.yaml", overrides + [f"task.test_digits={list(LENGTHS)}", f"task.test_examples={EXAMPLES}"])
    task = Arithmetic(cfg.task, 99, device)
    model = Transformer(16, cfg.model).to(device).eval()
    return cfg, task, model


def checkpoints(run_dir: Path):
    return sorted(run_dir.glob("step_*.pt"), key=lambda f: int(f.stem[5:]))


@torch.no_grad()
def column_mass(model, ex, n):
    """Mean attention from the token before each answer digit onto the two digits of
    that column, for the head that puts the most mass there at the training length."""
    for block in model.blocks:
        block.attn.record = True
    model(ex["tokens"], ex["positions"])
    p = ex["prompt_len"]
    mass = []
    for block in model.blocks:
        pat = block.attn.pattern.float().mean(0)
        block.attn.record = False
        # Answer digit i is predicted at the token before it; its column is a_i and b_i.
        per_head = sum(pat[:, p + i - 1, [1 + i, 2 + n + i]].sum(-1) for i in range(n))
        mass.append(per_head / n)
    return torch.stack(mass)


@torch.no_grad()
def exact_match(model, task, ex):
    return task.accuracy(model, ex)


def set_scale(model, s):
    for block in model.blocks:
        block.attn.scale = s


if __name__ == "__main__":
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Panel 1: sharpness and accuracy against length for the final checkpoint of each run.
    ax = axes[0]
    adder = {}
    for color, (label, pattern) in zip(colors, RUNS.items()):
        for seed in SEEDS:
            run_dir = Path(pattern.format(seed))
            if not run_dir.exists():
                continue
            cfg, task, model = load(run_dir)
            model.load_state_dict(torch.load(checkpoints(run_dir)[-1], map_location=device))
            masses = torch.stack([column_mass(model, task.test_sets[n], n) for n in LENGTHS])
            head = masses[LENGTHS.index(20)].flatten().argmax()
            adder[(label, seed)] = (int(head) // cfg.model.n_heads, int(head) % cfg.model.n_heads)
            sharp = masses.flatten(1)[:, head].cpu().numpy()
            acc = [exact_match(model, task, task.test_sets[n]) for n in LENGTHS]
            ax.plot(LENGTHS, sharp, color=color, marker="o", ms=3, label=label if seed == 0 else None)
            ax.plot(LENGTHS, acc, color=color, ls=":", lw=1)
    ax.set(xlabel="test digits", ylabel="fraction", ylim=(-0.03, 1.03))
    ax.set_title("adder head mass on its column (solid),\nexact match (dotted)", loc="left", fontsize=9.5)
    ax.legend(fontsize=8)

    # Panel 2: sharpness at 20 digits across training, with 40-digit accuracy from the logs.
    ax = axes[1]
    for color, (label, pattern) in zip(colors, RUNS.items()):
        for seed in SEEDS:
            run_dir = Path(pattern.format(seed))
            if not run_dir.exists():
                continue
            cfg, task, model = load(run_dir)
            layer, h = adder[(label, seed)]
            steps, sharp = [], []
            for ckpt in checkpoints(run_dir)[1:]:
                model.load_state_dict(torch.load(ckpt, map_location=device))
                steps.append(int(ckpt.stem[5:]))
                sharp.append(column_mass(model, task.test_sets[20], 20)[layer, h].item())
            m = read_metrics(run_dir)
            ax.plot(steps, sharp, color=color, marker="o", ms=3, label=label if seed == 0 else None)
            ax.plot(m["step"], m["acc_40"], color=color, ls=":", lw=1)
    ax.set(xlabel="training step", ylim=(-0.03, 1.03))
    ax.set_title("same at 20 digits over training (solid),\n40-digit exact match (dotted)", loc="left", fontsize=9.5)

    # Panel 3: inference-time sharpening on the final weight decay checkpoints.
    ax = axes[2]
    label, pattern = next(iter(RUNS.items()))
    for si, s in enumerate(SCALES):
        curves = []
        for seed in SEEDS:
            run_dir = Path(pattern.format(seed))
            cfg, task, model = load(run_dir)
            model.load_state_dict(torch.load(checkpoints(run_dir)[-1], map_location=device))
            set_scale(model, s)
            curves.append([exact_match(model, task, task.test_sets[n]) for n in LENGTHS])
            set_scale(model, 1.0)
        ax.plot(LENGTHS, np.mean(curves, 0), color=colors[si], marker="o", ms=3, label=f"logits x {s}")
        print(f"scale {s}: " + " ".join(f"{n}:{v:.2f}" for n, v in zip(LENGTHS, np.mean(curves, 0))))
    ax.set(xlabel="test digits", ylim=(-0.03, 1.03))
    ax.set_title(f"{label}, attention sharpened at inference\n(mean of 3 seeds)", loc="left", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "attention_dilution")
