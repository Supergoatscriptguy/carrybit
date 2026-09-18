"""Exact match against test length for the 11M position coupling models when the attention
logits are scaled at inference. Trained on 1 to 30 digits, evaluated to 200.

    uv run python experiments/sharpening_sweep.py

Generation at 200 digits without a key-value cache is slow; this takes a few minutes.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from carrybit.arithmetic import Arithmetic
from carrybit.config import load_config
from carrybit.model import Transformer
from carrybit.plotting import save, use_style

torch.cuda.set_per_process_memory_fraction(0.85)  # see train.py
device = "cuda"
RUNS = {1: "runs/addition_big/position_coupling_s1", 0: "runs/addition_big/position_coupling_s0"}
LENGTHS = (10, 30, 40, 60, 80, 100, 120, 150, 200)
SCALES = (1.0, 1.4, 1.6, 2.0, 2.5)
EXAMPLES = 128


def evaluate(run_dir: Path):
    cache = run_dir / "sharpening.csv"
    if cache.exists():
        with open(cache) as f:
            rows = list(csv.DictReader(f))
        return {float(r["scale"]): [float(r[str(n)]) for n in LENGTHS] for r in rows}
    raw = json.loads((run_dir / "config.json").read_text())
    overrides = [f"{s}.{k}={json.dumps(v)}" for s in ("task", "model") for k, v in raw[s].items() if k not in ("kind", "test_digits")]
    cfg = load_config("configs/addition_big.yaml", overrides + [f"task.test_digits={list(LENGTHS)}", f"task.test_examples={EXAMPLES}"])
    task = Arithmetic(cfg.task, 99, device)
    model = Transformer(16, cfg.model).to(device).eval()
    last = max(run_dir.glob("step_*.pt"), key=lambda f: int(f.stem[5:]))
    model.load_state_dict(torch.load(last, map_location=device))
    results = {}
    for scale in SCALES:
        for block in model.blocks:
            block.attn.scale = scale
        results[scale] = [task.accuracy(model, task.test_sets[n]) for n in LENGTHS]
        torch.cuda.empty_cache()
        print(f"{run_dir.name} x{scale}: " + " ".join(f"{n}:{a:.2f}" for n, a in zip(LENGTHS, results[scale])), flush=True)
    with open(cache, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", *LENGTHS])
        writer.writerows([scale, *acc] for scale, acc in results.items())
    return results


if __name__ == "__main__":
    use_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for seed, run in RUNS.items():
        results = evaluate(Path(run))
        # Strong scales are drawn first so the milder ones stay visible where they overlap.
        for color, (scale, acc) in reversed(list(zip(colors, results.items()))):
            ax.plot(LENGTHS, acc, color=color, marker="o", ms=3, lw=1.8 if seed == 1 else 1, alpha=1 if seed == 1 else 0.35,
                    label=f"logits x {scale}" if seed == 1 else None)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=8, loc="lower left")
    ax.axvline(30, color="0.6", lw=0.8, ls=":")
    ax.text(32, 0.5, "trained up to 30", color="0.4", fontsize=8)
    ax.set(xlabel="test digits", ylabel="exact match", ylim=(-0.03, 1.03))
    ax.set_title("11M position coupling, attention sharpened at inference\n(bold: seed 1, faint: seed 0)", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "sharpening_sweep")
