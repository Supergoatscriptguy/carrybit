"""Train and test accuracy over training for the modular addition run.

    uv run python -m carrybit.train configs/modular_add.yaml
    uv run python experiments/grokking_curve.py
"""

import sys

import matplotlib.pyplot as plt

from carrybit.plotting import read_metrics, save, use_style

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/modular_add"
m = read_metrics(run_dir)

use_style()
fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(9, 3.4))
ax_acc.plot(m["step"], m["train_acc"], label="train")
ax_acc.plot(m["step"], m["test_acc"], label="test")
ax_acc.set(xlabel="step", ylabel="accuracy", xscale="log", ylim=(0, 1.02))
ax_acc.legend()

ax_loss.plot(m["step"], m["train_loss"], label="train")
ax_loss.plot(m["step"], m["test_loss"], label="test")
ax_loss.set(xlabel="step", ylabel="loss", xscale="log", yscale="log")
fig.suptitle("Modular addition mod 113: grokking", x=0.02, ha="left")
save(fig, "grokking_curve")
