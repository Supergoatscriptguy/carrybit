import torch
import torch.nn.functional as F

from carrybit.config import ModularConfig


class ModularAddition:
    """a + b mod p as the sequence [a, b, =], with the answer read off at the '=' position.
    Every pair appears exactly once, split into a fixed train and test set."""

    def __init__(self, cfg: ModularConfig, seed: int, device):
        self.p = cfg.p
        self.eq = cfg.p
        self.vocab_size = cfg.p + 1
        self.device = device

        a, b = torch.meshgrid(torch.arange(cfg.p), torch.arange(cfg.p), indexing="ij")
        pairs = torch.stack([a.flatten(), b.flatten()], dim=1)
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(pairs), generator=gen)
        n_train = int(cfg.train_frac * len(pairs))
        self.train = self._encode(pairs[perm[:n_train]])
        self.test = self._encode(pairs[perm[n_train:]])

    def _encode(self, pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        tokens = torch.stack([a, b, torch.full_like(a, self.eq)], dim=1)
        targets = torch.full_like(tokens, -100)
        targets[:, -1] = (a + b) % self.p
        return tokens.to(self.device), targets.to(self.device)

    def train_batch(self, batch_size: int):
        tokens, targets = self.train
        if batch_size == 0:
            return tokens, targets, None
        idx = torch.randint(len(tokens), (batch_size,), device=self.device)
        return tokens[idx], targets[idx], None

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        out = {}
        for split, (tokens, targets) in (("train", self.train), ("test", self.test)):
            logits = model(tokens)[:, -1]
            out[f"{split}_loss"] = F.cross_entropy(logits, targets[:, -1]).item()
            out[f"{split}_acc"] = (logits.argmax(-1) == targets[:, -1]).float().mean().item()
        return out
