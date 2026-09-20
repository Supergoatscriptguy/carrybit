"""Scale the attention logits at inference on every model that ever generalized past its
training length, and measure exact match against test length. Two figures: the full
curve for the 11M position coupling models, and the "reach" (longest length with at least
90% exact match) of every run before and after sharpening at its best scale.

    uv run python experiments/sharpening_sweep.py

Results are cached in each run's sharpening.csv. Generation at 200 digits without a
key-value cache is slow; a cold run takes a while.
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
SCALES = (1.0, 1.2, 1.4, 1.6, 2.0, 2.5)
SMALL = (20, 30, 40, 50, 60, 80, 100)
BIG = (30, 40, 60, 80, 100, 120, 150, 200)
RUNS = [
    ("11M coupling", "configs/addition_big.yaml", "runs/addition_big/position_coupling_s{}", (0, 1, 2, 3), BIG),
    ("11M abacus", "configs/addition_big.yaml", "runs/addition_big/abacus_s{}", (0, 1), BIG),
    ("11M blankspace fixed", "configs/blankspace_big.yaml", "runs/blankspace_big/blankspace_fixed_s{}", (0, 1), SMALL),
    ("3M coupling", "configs/addition.yaml", "runs/addition/position_coupling_s{}", (0, 1, 2, 3, 4, 5), SMALL),
    ("3M coupling, no wd", "configs/addition.yaml", "runs/addition_no_wd/position_coupling_s{}", (0, 1, 2, 3, 4, 5), SMALL),
    ("3M abacus", "configs/addition.yaml", "runs/addition/abacus_s{}", (0, 1, 2), SMALL),
    ("3M blankspace fixed", "configs/addition.yaml", "runs/addition/blankspace_fixed_s{}", (0, 1, 2), SMALL),
]
EXAMPLES = 128


def evaluate(run_dir: Path, config: str, lengths):
    cache = run_dir / "sharpening.csv"
    if cache.exists():
        with open(cache) as f:
            rows = list(csv.DictReader(f))
        if set(rows[0]) == {"scale", *map(str, lengths)} and len(rows) == len(SCALES):
            return {float(r["scale"]): [float(r[str(n)]) for n in lengths] for r in rows}
    raw = json.loads((run_dir / "config.json").read_text())
    overrides = [f"{s}.{k}={json.dumps(v)}" for s in ("task", "model") for k, v in raw[s].items() if k not in ("kind", "test_digits")]
    cfg = load_config(config, overrides + [f"task.test_digits={list(lengths)}", f"task.test_examples={EXAMPLES}"])
    task = Arithmetic(cfg.task, 99, device)
    model = Transformer(16, cfg.model).to(device).eval()
    last = max(run_dir.glob("step_*.pt"), key=lambda f: int(f.stem[5:]))
    model.load_state_dict(torch.load(last, map_location=device))
    results = {}
    for scale in SCALES:
        for block in model.blocks:
            block.attn.scale = scale
        results[scale] = [task.accuracy(model, task.test_sets[n]) for n in lengths]
        torch.cuda.empty_cache()
        print(f"{run_dir} x{scale}: " + " ".join(f"{n}:{a:.2f}" for n, a in zip(lengths, results[scale])), flush=True)
    with open(cache, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", *lengths])
        writer.writerows([scale, *acc] for scale, acc in results.items())
    return results


def curve_figure(results_by_seed, lengths):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] + ["0.4"]
    for seed, results in results_by_seed.items():
        bold = seed == 1
        # Strong scales are drawn first so the milder ones stay visible where they overlap.
        for color, (scale, acc) in reversed(list(zip(colors, results.items()))):
            ax.plot(lengths, acc, color=color, marker="o", ms=3, lw=1.8 if bold else 1, alpha=1 if bold else 0.3,
                    label=f"logits x {scale}" if bold else None)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=8, loc="lower left")
    ax.axvline(30, color="0.6", lw=0.8, ls=":")
    ax.text(32, 0.5, "trained up to 30", color="0.4", fontsize=8)
    ax.set(xlabel="test digits", ylabel="exact match", ylim=(-0.03, 1.03))
    ax.set_title("11M position coupling, attention sharpened at inference\n(bold: seed 1, faint: other seeds)", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "sharpening_sweep")


def reach_figure(rows):
    """rows: (label, {multiple: (before, after, best_scale)}) for multiples 2 and 3 of the
    training length. Exact match as trained versus at the best inference scale."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 0.26 * len(rows) + 1.2), sharey=True)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    for ax, multiple in zip(axes, (2, 3)):
        for y, (label, at) in enumerate(rows):
            before, after, best_scale = at[multiple]
            ax.plot([before, after], [y, y], color="0.75", lw=1, zorder=1)
            ax.plot(before, y, marker="o", mfc="white", mec="0.4", ms=5, zorder=2)
            ax.plot(after, y, marker="o", color=color, ms=5, zorder=3)
            if after - before >= 0.05:
                ax.text(after + 0.03, y, f"x{best_scale:g}", va="center", fontsize=7, color="0.4")
        ax.set(xlim=(-0.03, 1.15), xticks=(0, 0.5, 1), xlabel=f"exact match at {multiple}x the training length")
    axes[0].set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows], ylim=(-0.7, len(rows) - 0.3))
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0, labelsize=8)
    axes[1].plot([], [], marker="o", mfc="white", mec="0.4", ls="none", label="as trained")
    axes[1].plot([], [], marker="o", color=color, ls="none", label="best inference scale")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "sharpening_reach")


if __name__ == "__main__":
    use_style()
    rows = []
    coupling_curves = {}
    for label, config, pattern, seeds, lengths in RUNS:
        train_len = load_config(config).task.max_digits
        for seed in seeds:
            run_dir = Path(pattern.format(seed))
            if not (run_dir / "metrics.csv").exists():
                continue
            results = evaluate(run_dir, config, lengths)
            if label == "11M coupling":
                coupling_curves[seed] = results
            at = {}
            for multiple in (2, 3):
                n = min(lengths, key=lambda l: abs(l - multiple * train_len))
                i = lengths.index(n)
                best_scale = max(results, key=lambda s: (results[s][i], -s))
                at[multiple] = (results[1.0][i], results[best_scale][i], best_scale)
            rows.append((f"{label} s{seed}", at))
            print(f"{label} s{seed:<3} " + "  ".join(f"{m}x: {b:.2f} -> {a:.2f} (x{s:g})" for m, (b, a, s) in at.items()))
    curve_figure(coupling_curves, BIG)
    reach_figure(rows)
