"""Train every rung of the length generalization ladder and plot accuracy against test
digit count. Finished runs are skipped, so this can be interrupted and restarted.

    uv run python experiments/length_ladder.py            # train what is missing, then plot
    uv run python experiments/length_ladder.py --plot     # plot only
"""

import argparse
import json
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="skip training")
    if not ap.parse_args().plot:
        train_missing()
    plot()
