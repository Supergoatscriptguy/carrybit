from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    d_mlp: int = 512
    max_positions: int = 128
    positional: bool = True
    dropout: float = 0.0


@dataclass
class ModularConfig:
    p: int = 113
    train_frac: float = 0.3


@dataclass
class TrainConfig:
    steps: int = 40_000
    batch_size: int = 0  # 0 means full batch, which only makes sense for the modular task
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 0
    eval_every: int = 100
    checkpoint_every: int = 1000
    seed: int = 0
    wandb: bool = False


TASKS = {"modular": ModularConfig}


@dataclass
class Config:
    name: str
    task: ModularConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"]["kind"] = next(k for k, v in TASKS.items() if isinstance(self.task, v))
        return d


def _build(cls, values: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name not in values:
            continue
        v = values[f.name]
        if f.name == "task":
            kind = v.pop("kind")
            v = _build(TASKS[kind], v)
        elif is_dataclass(f.type):
            v = _build(f.type, v)
        elif isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def _set_nested(d: dict, dotted: str, value):
    *path, last = dotted.split(".")
    for key in path:
        d = d.setdefault(key, {})
    d[last] = value


def load_config(path: str | Path, overrides: list[str] = ()) -> Config:
    """Load a YAML config. Overrides look like train.lr=3e-4 and are parsed as YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        _set_nested(raw, key, yaml.safe_load(value))
    return _build(Config, raw)
