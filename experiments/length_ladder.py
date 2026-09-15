"""Train every rung of the length generalization ladder and plot accuracy against test
digit count. Finished runs are skipped, so this can be interrupted and restarted.

    uv run python experiments/length_ladder.py            # train what is missing, then plot
    uv run python experiments/length_ladder.py --plot     # plot only
    uv run python experiments/length_ladder.py --watch    # live status board for a running ladder
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from carrybit.config import load_config
from carrybit.plotting import read_metrics, save, use_style
from carrybit.train import train

RUNGS = {
    "plain": ["task.reverse=false"],
    "reversed": [],
    "reversed + zero pad": ["task.zero_pad=true"],
    "abacus": ["task.positions=abacus"],
    "position coupling": ["task.zero_pad=true", "task.positions=coupled"],
    "aligned blankspace": ["task.zero_pad=true", "task.blanks=100"],
}
SEEDS = (0, 1, 2)
RUNS = Path("runs/ladder")


def run_dir(rung: str, seed: int) -> Path:
    return RUNS / f"{rung.replace(' + ', '_').replace(' ', '_')}_s{seed}"


def finished(d: Path) -> bool:
    if not (d / "metrics.csv").exists():
        return False
    steps = json.loads((d / "config.json").read_text())["train"]["steps"]
    return int(read_metrics(d)["step"][-1]) == steps


def train_missing():
    for rung, overrides in RUNGS.items():
        for seed in SEEDS:
            d = run_dir(rung, seed)
            if finished(d):
                continue
            name = f"ladder {rung} seed {seed}"
            cfg = load_config("configs/addition.yaml", [*overrides, f"train.seed={seed}", f"name={name}"])
            train(cfg, d)
            torch.cuda.empty_cache()


def plot():
    use_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    train_digits = load_config("configs/addition.yaml").task.max_digits
    for color, rung in zip(plt.rcParams["axes.prop_cycle"].by_key()["color"] * 2, RUNGS):
        curves = []
        for seed in SEEDS:
            d = run_dir(rung, seed)
            if not finished(d):
                continue
            m = read_metrics(d)
            digits = sorted(int(k[4:]) for k in m if k.startswith("acc_"))
            curves.append([m[f"acc_{n}"][-1] for n in digits])
            ax.plot(digits, curves[-1], color=color, alpha=0.25, lw=1)
        if curves:
            style = "--" if rung == "abacus" else "-"
            ax.plot(digits, np.mean(curves, 0), color=color, ls=style, label=rung)
    ax.axvline(train_digits, color="0.6", lw=0.8, ls=":")
    ax.text(train_digits + 1, 0.02, "train length", color="0.4", fontsize=8)
    ax.set(xlabel="test digits", ylabel="exact match accuracy", ylim=(-0.02, 1.02))
    ax.legend(loc="upper right")
    save(fig, "length_ladder")


def watch(refresh: float = 10.0):
    steps = load_config("configs/addition.yaml").train.steps
    while True:
        lines, done = [], 0
        for rung in RUNGS:
            for seed in SEEDS:
                d = run_dir(rung, seed)
                label = f"{rung:<20} seed {seed}"
                if not (d / "metrics.csv").exists():
                    lines.append(f"  {label}  pending")
                    continue
                m = read_metrics(d)
                acc = " ".join(f"{k[4:]}:{m[k][-1]:.2f}" for k in m if k.startswith("acc_"))
                if int(m["step"][-1]) == steps:
                    done += 1
                    lines.append(f"  {label}  done   {acc}")
                    continue
                # Metrics only land at eval time, so extrapolate the step from the wall clock.
                last_step, last_elapsed = m["step"][-1], m["elapsed"][-1]
                started = os.path.getmtime(d / "metrics.csv") - last_elapsed
                rate = last_step / last_elapsed if last_step else 0
                est = min(steps, int((time.time() - started) * rate)) if rate else 0
                remaining = (steps - est) / rate / 60 if rate else float("nan")
                bar = "#" * int(30 * est / steps)
                lines.append(f"> {label}  [{bar:<30}] {est:>6}/{steps}  ~{remaining:.0f} min left")
                lines.append(f"  {'':<27}{acc}")
        os.system("cls" if os.name == "nt" else "clear")
        print(f"length ladder  {done}/{len(RUNGS) * len(SEEDS)} runs done  ({time.strftime('%H:%M:%S')})\n")
        print("\n".join(lines))
        time.sleep(refresh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="skip training")
    ap.add_argument("--watch", action="store_true", help="show live progress of a running ladder")
    args = ap.parse_args()
    if args.watch:
        watch()
    elif args.plot:
        plot()
    else:
        train_missing()
        plot()
