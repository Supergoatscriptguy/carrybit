"""Out-of-distribution accuracy as a function of training step, for runs that generalized
past their training length at some point. Reads the metrics the ladder runs already logged.

    uv run python experiments/generalization_over_training.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from carrybit.plotting import read_metrics, save, use_style

PANELS = [
    ("position coupling, weight decay 0.1", "runs/addition/position_coupling_s{}", (0, 1, 2), 40),
    ("same, constant learning rate", "runs/addition_constant_lr/position_coupling_s{}", (0, 1, 2), 40),
    ("same, no weight decay", "runs/addition_no_wd/position_coupling_s{}", (0, 1, 2), 40),
    ("11M params, trained to 30 digits", "runs/addition_big/position_coupling_s{}", (0, 1), 60),
]

use_style()
fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for ax, color, (title, pattern, seeds, digits) in zip(axes.flat, palette, PANELS):
    for seed in seeds:
        run = Path(pattern.format(seed))
        if not run.exists():
            continue
        m = read_metrics(run)
        train_len = max(int(k[4:]) for k in m if k.startswith("acc_") and int(k[4:]) <= 30)
        ax.plot(m["step"], m[f"acc_{train_len}"], color="0.7", lw=1, ls=":")
        ax.plot(m["step"], m[f"acc_{digits}"], color=color, lw=1.5, alpha=0.9)
    ax.set_title(f"{title}: {digits} digits", loc="left", fontsize=9.5)
    ax.set(ylim=(-0.03, 1.03))
for ax in axes[1]:
    ax.set_xlabel("training step")
for ax in axes[:, 0]:
    ax.set_ylabel("exact match")
axes[1, 0].text(0.98, 0.08, "dotted: longest training length", transform=axes[1, 0].transAxes, ha="right", fontsize=8, color="0.4")
fig.tight_layout()
save(fig, "generalization_over_training")
