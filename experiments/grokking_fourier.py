"""Track Fourier structure in the weights across training, reproducing the measurement of
Swaroop (2026) on their ReLU MLP and applying it to the one-layer transformer.

    uv run python -m carrybit.train configs/modular_mlp.yaml
    uv run python -m carrybit.train configs/modular_add.yaml
    uv run python -m carrybit.train configs/modular_add.yaml train.weight_decay=0.1 name=modular_add_wd0.1 --run-dir runs/modular_add_wd0.1
    uv run python -m carrybit.train configs/modular_add.yaml train.weight_decay=0 name=modular_add_wd0 --run-dir runs/modular_add_wd0
    uv run python experiments/grokking_fourier.py

For the transformer, a neuron's "input weights" are its pre-activation averaged over the
other operand, and its "output weights" are its column of the down projection pushed
through the unembedding, ignoring the final layer norm.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from carrybit.config import load_config
from carrybit.fourier import analyze_neurons
from carrybit.model import build_model
from carrybit.plotting import read_metrics, save, use_style

TRANSFORMER_RUNS = {"weight decay 1": "runs/modular_add", "weight decay 0.1": "runs/modular_add_wd0.1", "no weight decay": "runs/modular_add_wd0"}


def load(run_dir: Path):
    raw = json.loads((run_dir / "config.json").read_text())
    cfg = load_config("configs/modular_add.yaml", [f"model.{k}={json.dumps(v)}" for k, v in raw["model"].items()])
    cfg.task.p = raw["task"]["p"]
    return cfg, build_model(cfg.task.p + 1, cfg.model).eval()


def mlp_weights(model, p):
    w = model.hidden.weight.detach()
    return w[:, :p], w[:, p + 1 : 2 * p + 1], model.out.weight.detach()[:p].T


@torch.no_grad()
def transformer_weights(model, p):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    tokens = torch.stack([a.flatten(), b.flatten(), torch.full((p * p,), p)], 1)
    pre = {}
    handle = model.blocks[0].mlp[0].register_forward_hook(lambda m, i, o: pre.__setitem__("x", o[:, -1]))
    model(tokens)
    handle.remove()
    f = pre["x"].view(p, p, -1)
    w_out = model.unembed.weight[:p] @ model.blocks[0].mlp[2].weight
    return f.mean(1).T, f.mean(0).T, w_out.T


def track(run_dir: Path) -> dict[str, np.ndarray]:
    out = run_dir / "fourier.csv"
    if not out.exists():
        cfg, model = load(run_dir)
        p = cfg.task.p
        metrics = read_metrics(run_dir)
        rows = []
        for ckpt in sorted(run_dir.glob("step_*.pt"), key=lambda f: int(f.stem[5:])):
            step = int(ckpt.stem[5:])
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            weights = mlp_weights(model, p) if cfg.model.arch == "mlp" else transformer_weights(model, p)
            i = int(np.searchsorted(metrics["step"], step))
            rows.append({"step": step, "train_acc": metrics["train_acc"][i], "test_acc": metrics["test_acc"][i], **analyze_neurons(p, *weights)})
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")
    with open(out) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


PANELS = [
    ("test_acc", "accuracy", "test accuracy"),
    ("structured_frac", "fraction of neurons", "periodic neurons"),
    ("phase_alignment", "mean cos(residual)", "phase sum relation"),
    ("idealized_acc", "accuracy", "idealized model accuracy"),
]


def plot(curves: dict[str, dict], name: str, title: str):
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2), sharex=True)
    first = min(m["step"][m["step"] > 0].min() for m in curves.values())
    for ax, (key, ylabel, panel_title) in zip(axes.flat, PANELS):
        for label, m in curves.items():
            keep = m["step"] >= first
            x = m["step"][keep]
            ax.plot(x, m[key][keep], label=label)
            if key == "test_acc":
                ax.plot(x, m["train_acc"][keep], color=ax.lines[-1].get_color(), ls=":", lw=1)
            if key == "idealized_acc":
                ax.plot(x, m["test_acc"][keep], color=ax.lines[-1].get_color(), ls=":", lw=1)
        ax.set(xscale="log", title=panel_title, ylabel=ylabel, ylim=(-0.03, 1.03) if key != "phase_alignment" else (-1.03, 1.03))
    for ax in axes[1]:
        ax.set_xlabel("step")
    axes[0, 0].text(0.02, 0.9, "dotted: train", transform=axes[0, 0].transAxes, fontsize=8, color="0.4")
    axes[1, 1].text(0.02, 0.9, "dotted: the network's own test accuracy", transform=axes[1, 1].transAxes, fontsize=8, color="0.4")
    if len(curves) > 1:
        axes[0, 1].legend(loc="lower right")
    fig.suptitle(title, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    plot({"mlp": track(Path("runs/modular_mlp"))}, "grokking_fourier_mlp", "ReLU MLP, p = 97 (Swaroop 2026 setup)")
    plot({k: track(Path(v)) for k, v in TRANSFORMER_RUNS.items() if Path(v, "metrics.csv").exists()},
         "grokking_fourier_transformer", "One-layer transformer, p = 113")
