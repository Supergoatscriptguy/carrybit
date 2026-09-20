"""Which parts of the model need sharpening? On the 11M position coupling models, scale
the attention logits of (a) one layer at a time, (b) the first k layers, (c) every layer
but one, and (d) one head of layer 0 at a time, and measure exact match past the training
length. Compared with scaling nothing and scaling everything.

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


def sweep(run_dir: Path):
    cache = run_dir / "sharpen_by_layer.csv"
    if cache.exists():
        with open(cache) as f:
            return {r["what"]: [float(r[str(n)]) for n in LENGTHS] for r in csv.DictReader(f)}
    cfg, task, model = load(run_dir)
    L, H = cfg.model.n_layers, cfg.model.n_heads
    rows = {}

    def record(what, per_layer):
        for block, s in zip(model.blocks, per_layer):
            block.attn.scale = s
        rows[what] = [task.accuracy(model, task.test_sets[n]) for n in LENGTHS]
        for block in model.blocks:
            block.attn.scale = 1.0
        print(f"{run_dir.name} {what:14s} " + " ".join(f"{n}:{a:.2f}" for n, a in zip(LENGTHS, rows[what])), flush=True)

    record("none", [1.0] * L)
    record("all", [SCALE] * L)
    for layer in range(L):
        record(f"only {layer}", [SCALE if l == layer else 1.0 for l in range(L)])
    for k in range(1, L):
        record(f"first {k}", [SCALE if l < k else 1.0 for l in range(L)])
    for layer in range(L):
        record(f"all but {layer}", [1.0 if l == layer else SCALE for l in range(L)])
    for head in range(H):
        scales = torch.ones(H, device=device)
        scales[head] = SCALE
        record(f"L0H{head}", [scales if l == 0 else 1.0 for l in range(L)])
    with open(cache, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["what", *LENGTHS])
        writer.writerows([what, *acc] for what, acc in rows.items())
    return rows


if __name__ == "__main__":
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for seed, run in RUNS.items():
        rows = sweep(Path(run))
        L = sum(1 for w in rows if w.startswith("only"))
        H = sum(1 for w in rows if w.startswith("L0H"))
        style = dict(lw=1.8, alpha=1) if seed == 1 else dict(lw=1, alpha=0.35)
        for ax, (title, labels, keys) in zip(axes, [
            ("only this layer scaled", [str(l) for l in range(L)], [f"only {l}" for l in range(L)]),
            ("first k layers scaled", [str(k) for k in range(L + 1)], ["none"] + [f"first {k}" for k in range(1, L)] + ["all"]),
            ("every layer but this one scaled", [str(l) for l in range(L)], [f"all but {l}" for l in range(L)]),
        ]):
            for color, n in zip(colors, LENGTHS):
                ax.plot(labels, [rows[k][LENGTHS.index(n)] for k in keys], color=color, marker="o", ms=3,
                        label=f"{n} digits" if seed == 1 else None, **style)
            ax.set(ylim=(-0.03, 1.03))
            ax.set_title(title, loc="left", fontsize=9.5)
        print(f"seed {seed} layer 0 heads at {LENGTHS[0]} digits: " + " ".join(f"H{h}:{rows[f'L0H{h}'][0]:.2f}" for h in range(H)))
    axes[0].set(xlabel="layer", ylabel="exact match")
    axes[1].set(xlabel="k")
    axes[2].set(xlabel="layer left unscaled")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"11M position coupling, attention logits x{SCALE:g} in a subset of layers (bold: seed 1, faint: seed 0)",
                 x=0.02, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "sharpen_by_head")
