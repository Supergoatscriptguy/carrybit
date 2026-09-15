import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from carrybit.arithmetic import Arithmetic
from carrybit.config import ArithmeticConfig, Config, ModularConfig, load_config
from carrybit.model import Transformer
from carrybit.modular import ModularAddition


def make_task(cfg: Config, device):
    if isinstance(cfg.task, ModularConfig):
        return ModularAddition(cfg.task, cfg.train.seed, device)
    if isinstance(cfg.task, ArithmeticConfig):
        return Arithmetic(cfg.task, cfg.train.seed, device)
    raise ValueError(f"unknown task {cfg.task}")


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.cosine:
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr


def train(cfg: Config, run_dir: Path, device="cuda"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    torch.manual_seed(cfg.train.seed)

    task = make_task(cfg, device)
    model = Transformer(task.vocab_size, cfg.model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, betas=cfg.train.betas, weight_decay=cfg.train.weight_decay
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp)
    print(f"{cfg.name}: {model.n_params():,} params", flush=True)

    log = None
    if cfg.train.wandb:
        import wandb

        log = wandb.init(project="carrybit", name=cfg.name, config=cfg.to_dict())

    metrics_file = open(run_dir / "metrics.csv", "w", newline="")
    writer = None
    t0 = time.time()
    loss_ema = float("nan")
    for step in range(cfg.train.steps + 1):
        last = step == cfg.train.steps
        if step % cfg.train.eval_every == 0 or last:
            model.eval()
            with autocast:
                metrics = task.evaluate(model)
            row = {"step": step, "elapsed": round(time.time() - t0, 1), "loss": loss_ema, **metrics}
            model.train()
            if writer is None:
                writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            metrics_file.flush()
            if log:
                log.log(row, step=step)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if step % cfg.train.checkpoint_every == 0 or last:
            torch.save(model.state_dict(), run_dir / f"step_{step}.pt")
        if last:
            break

        tokens, targets, positions = task.train_batch(cfg.train.batch_size)
        with autocast:
            logits = model(tokens, positions)
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
        loss_ema = loss.item() if math.isnan(loss_ema) else 0.98 * loss_ema + 0.02 * loss.item()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg.train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    metrics_file.close()
    if log:
        log.finish()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="dotted overrides like train.lr=3e-4")
    ap.add_argument("--run-dir", help="defaults to runs/<name>")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    assert torch.cuda.is_available(), "no CUDA device found"
    train(cfg, Path(args.run_dir or f"runs/{cfg.name}"))


if __name__ == "__main__":
    main()
