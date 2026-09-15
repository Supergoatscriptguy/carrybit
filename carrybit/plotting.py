import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STYLE = ROOT / "assets" / "carrybit.mplstyle"
FIGURES = ROOT / "assets" / "figures"


def use_style():
    plt.style.use(STYLE)


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path.relative_to(ROOT)}")


def read_metrics(run_dir: str | Path) -> dict[str, np.ndarray]:
    with open(Path(run_dir) / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
