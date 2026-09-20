"""Which heads need sharpening? On the 11M position coupling models, scale the attention
logits of one layer at a time, then one head at a time, and measure exact match past the
training length. Compared with scaling every head at once.

    uv run python experiments/sharpen_by_head.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from carrybit.arithmetic import Arithmetic
from carrybit.config import load_config
from carrybit.model import Transformer
from carrybit.plotting import save, use_style

torch.cuda.set_per_process_memory_fraction(0.85)  # see train.py
device = "cuda"
RUNS = {1: "runs/addition_big/position_coupling_s1", 0: "runs/addition_big/position_coupling_s0"}
LENGTHS = (60, 100, 150)
SCALE = 2.0
EXAMPLES = 64


def load(run_dir: Path):
    raw = json.loads((run_dir / "config.json").read_text())
    overrides = [f"{s}.{k}={json.dumps(v)}" for s in ("task", "model") for k, v in raw[s].items() if k not in ("kind", "test_digits")]
    cfg = load_config("configs/addition_big.yaml", overrides + [f"task.test_digits={list(LENGTHS)}", f"task.test_examples={EXAMPLES}"])
    task = Arithmetic(cfg.task, 99, device)
    model = Transformer(16, cfg.model).to(device).eval()
    last = max(run_dir.glob("step_*.pt"), key=lambda f: int(f.stem[5:]))
    model.load_state_dict(torch.load(last, map_location=device))
    return cfg, task, model


def set_scales(model, per_layer):
    """per_layer: list of floats or (n_heads,) tensors, one per block."""
    for block, s in zip(model.blocks, per_layer):
        block.attn.scale = s


def accuracy(model, task):
    return [task.accuracy(model, task.test_sets[n]) for n in LENGTHS]


def sweep(run_dir: Path):
    cache = run_dir / "sharpen_by_head.csv"
    if cache.exists():
        with open(cache) as f:
            return [(r["what"], [float(r[str(n)]) for n in LENGTHS]) for r in csv.DictReader(f)]
    cfg, task, model = load(run_dir)
    L, H = cfg.model.n_layers, cfg.model.n_heads
    rows = []

    def record(what, per_layer):
        set_scales(model, per_layer)
        acc = accuracy(model, task)
        set_scales(model, [1.0] * L)
        rows.append((what, acc))
        print(f"{run_dir.name} {what:12s} " + " ".join(f"{n}:{a:.2f}" for n, a in zip(LENGTHS, acc)), flush=True)

    record("none", [1.0] * L)
    record("all", [SCALE] * L)
    for layer in range(L):
        record(f"layer {layer}", [SCALE if l == layer else 1.0 for l in range(L)])
    for layer in range(L):
        for head in range(H):
            scales = torch.ones(H, device=device)
            scales[head] = SCALE
            record(f"L{layer}H{head}", [scales if l == layer else 1.0 for l in range(L)])
    with open(cache, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["what", *LENGTHS])
        writer.writerows([what, *acc] for what, acc in rows)
    return rows


if __name__ == "__main__":
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, (seed, run) in zip(axes, RUNS.items()):
        rows = sweep(Path(run))
        n_layers = sum(1 for what, _ in rows if what.startswith("layer"))
        n_heads = sum(1 for what, _ in rows if what.startswith("L0H"))
        grid = np.full((n_layers, n_heads), np.nan)
        for what, acc in rows:
            if what.startswith("L") and "H" in what:
                l, h = what[1:].split("H")
                grid[int(l), int(h)] = acc[LENGTHS.index(100)]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="Oranges", aspect="auto")
        for l in range(n_layers):
            for h in range(n_heads):
                ax.text(h, l, f"{grid[l, h]:.2f}", ha="center", va="center", fontsize=7, color="black" if grid[l, h] < 0.6 else "white")
        by_layer = {what: acc[LENGTHS.index(100)] for what, acc in rows}
        ax.set(xticks=range(n_heads), xticklabels=[f"H{h}" for h in range(n_heads)],
               yticks=range(n_layers), yticklabels=[f"L{l}  ({by_layer[f'layer {l}']:.2f})" for l in range(n_layers)])
        ax.set_title(f"seed {seed}: none {by_layer['none']:.2f}, all {by_layer['all']:.2f}", loc="left", fontsize=9.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(f"exact match at 100 digits when only one head (cells) or one layer (row labels) is scaled by {SCALE:g}",
                 x=0.02, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "sharpen_by_head")
