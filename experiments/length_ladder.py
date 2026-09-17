"""Train every rung of the length generalization ladder and plot accuracy against test
digit count. Finished runs are skipped, so this can be interrupted and restarted.

    uv run python experiments/length_ladder.py            # train what is missing, then plot
    uv run python experiments/length_ladder.py --plot     # plot only
    uv run python experiments/length_ladder.py --watch    # live status board for a running ladder

Pass --config to run the same rungs on another base config, and --rungs to pick a subset.
Runs land in runs/<config name> and the figure is named after the config. --set applies extra
config overrides and --name gives the resulting runs their own folder and figure, so an
ablation like --set train.cosine=false --name addition_constant_lr needs no new config file.
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
    "zero pad + relative": ["task.zero_pad=true", "model.pos_embed=relative"],
    "blankspace var": ["task.zero_pad=true", "task.blanks=121", "task.blanks_fixed=false"],
    "blankspace fixed": ["task.zero_pad=true", "task.blanks=121"],
    "blankspace fixed + relative": ["task.zero_pad=true", "task.blanks=121", "model.pos_embed=relative"],
}
SEEDS = (0, 1, 2)


def run_dir(base: str, rung: str, seed: int) -> Path:
    return Path("runs") / base / f"{rung.replace(' + ', '_').replace(' ', '_')}_s{seed}"


def finished(d: Path) -> bool:
    if not (d / "metrics.csv").exists():
        return False
    steps = json.loads((d / "config.json").read_text())["train"]["steps"]
    return int(read_metrics(d)["step"][-1]) == steps


def train_missing(config: str, rungs: list[str], seeds=SEEDS, overrides=()):
    base = load_config(config, overrides).name
    for rung in rungs:
        for seed in seeds:
            d = run_dir(base, rung, seed)
            if finished(d):
                continue
            name = f"{base} {rung} seed {seed}"
            cfg = load_config(config, [*overrides, *RUNGS[rung], f"train.seed={seed}", f"name={name}"])
            train(cfg, d)
            torch.cuda.empty_cache()


def plot(config: str, rungs: list[str], seeds=SEEDS, overrides=()):
    cfg = load_config(config, overrides)
    use_style()
    ncols = min(3, len(rungs))
    nrows = -(-len(rungs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.6 * nrows), sharex=True, sharey=True, squeeze=False)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {rung: palette[i % len(palette)] for i, rung in enumerate(RUNGS)}
    for ax in axes.flat[len(rungs):]:
        ax.set_visible(False)
    max_digits = max(cfg.task.test_digits)
    for ax, rung in zip(axes.flat, rungs):
        color = colors[rung]
        curves = []
        for seed in seeds:
            d = run_dir(cfg.name, rung, seed)
            if not finished(d):
                continue
            m = read_metrics(d)
            digits = sorted(int(k[4:]) for k in m if k.startswith("acc_"))
            curves.append([m[f"acc_{n}"][-1] for n in digits])
            ax.plot(digits, curves[-1], color=color, alpha=0.3, lw=1)
        if curves:
            ax.plot(digits, np.mean(curves, 0), color=color, lw=2)
        ax.axvline(cfg.task.max_digits, color="0.6", lw=0.8, ls=":")
        ax.set_title(rung, loc="left")
        ax.set(ylim=(-0.03, 1.03), xlim=(0, max_digits * 1.02))
    for ax in axes[-1]:
        ax.set_xlabel("test digits")
    for ax in axes[:, 0]:
        ax.set_ylabel("exact match")
    axes[0, 0].text(cfg.task.max_digits + max_digits / 50, 0.45, f"trained up to {cfg.task.max_digits}", color="0.4", fontsize=8)
    fig.tight_layout()
    suffix = cfg.name.partition("_")[2]
    save(fig, "length_ladder" + (f"_{suffix}" if suffix else ""))


def watch(config: str, rungs: list[str], seeds=SEEDS, overrides=(), refresh: float = 10.0):
    cfg = load_config(config, overrides)
    steps = cfg.train.steps
    while True:
        lines, done = [], 0
        for rung in rungs:
            for seed in seeds:
                d = run_dir(cfg.name, rung, seed)
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
    ap.add_argument("--config", default="configs/addition.yaml")
    ap.add_argument("--rungs", default=",".join(RUNGS), help="comma-separated subset of rungs")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra config override")
    ap.add_argument("--name", help="run folder and figure name when using --set")
    ap.add_argument("--plot", action="store_true", help="skip training")
    ap.add_argument("--watch", action="store_true", help="show live progress of a running ladder")
    args = ap.parse_args()
    rungs = args.rungs.split(",")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    overrides = args.set + ([f"name={args.name}"] if args.name else [])
    if args.watch:
        watch(args.config, rungs, seeds, overrides)
    elif args.plot:
        plot(args.config, rungs, seeds, overrides)
    else:
        train_missing(args.config, rungs, seeds, overrides)
        plot(args.config, rungs, seeds, overrides)
