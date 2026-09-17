import torch
import torch.nn.functional as F

from carrybit.config import ArithmeticConfig
from carrybit.tokenizer import BLANK, END, EQ, PAD, PLUS


def add_digits(a, b):
    """Ripple-carry addition of two (B, D) digit tensors stored least significant first.
    Returns the (B, D+1) sum digits and the (B, D) carry into each position."""
    B, D = a.shape
    total = torch.zeros(B, D + 1, dtype=a.dtype, device=a.device)
    carries = torch.zeros(B, D, dtype=a.dtype, device=a.device)
    carry = torch.zeros(B, dtype=a.dtype, device=a.device)
    for i in range(D):
        carries[:, i] = carry
        t = a[:, i] + b[:, i] + carry
        total[:, i] = t % 10
        carry = t // 10
    total[:, D] = carry
    return total, carries


def digit_count(digits):
    idx = torch.arange(1, digits.shape[1] + 1, device=digits.device)
    return ((digits != 0) * idx).amax(1).clamp(min=1)


def strip_blanks(tokens):
    """Move blank tokens to the end of each row, keeping the order of everything else."""
    order = (tokens == BLANK).long().argsort(dim=1, stable=True)
    return tokens.gather(1, order)


class Arithmetic:
    """Multi-digit addition with on-the-fly examples, laid out as $a+b=c$ and padded.
    All formatting is done with tensor ops so a batch costs about as much as a forward pass."""

    def __init__(self, cfg: ArithmeticConfig, seed: int, device):
        if cfg.positions == "coupled" and not (cfg.reverse and cfg.zero_pad):
            raise ValueError("coupled positions need reversed, zero-padded operands")
        if cfg.blanks and not cfg.zero_pad:
            raise ValueError("aligned blanks need zero-padded operands")
        if cfg.blanks and cfg.blanks < cfg.max_digits + 2:
            raise ValueError("blanks must leave room for max_digits + 1 digits per number")
        self.cfg = cfg
        self.device = device
        self.vocab_size = 16
        self.gen = torch.Generator(device=device).manual_seed(seed)
        # Test sets are fixed for the life of the task so every eval sees the same problems.
        eval_gen = torch.Generator(device=device).manual_seed(10_000 + seed)
        self.test_sets = {
            n: self.build(*self.sample(cfg.test_examples, n, n, gen=eval_gen), train=False)
            for n in cfg.test_digits
        }

    def sample(self, batch: int, max_digits: int, min_digits: int = 1, gen=None):
        """Random operands with lengths uniform in [min_digits, max_digits]. Digits are stored
        least significant first, so a shorter number is just one with trailing zeros."""
        gen = gen or self.gen
        D = max_digits
        lengths = torch.randint(min_digits, D + 1, (2, batch), device=self.device, generator=gen)
        digits = torch.randint(0, 10, (2, batch, D), device=self.device, generator=gen)
        idx = torch.arange(D, device=self.device)
        digits[idx >= lengths[..., None]] = 0
        # Leading digit must not be zero unless the number is a single digit.
        lead = torch.randint(1, 10, (2, batch), device=self.device, generator=gen)
        leading = idx == (lengths - 1)[..., None]
        digits = torch.where(leading & (lengths[..., None] > 1) & (digits == 0), lead[..., None], digits)
        return digits[0], digits[1], lengths[0], lengths[1]

    def build(self, a, b, la, lb, train: bool):
        """Turn digit tensors into token, target and position tensors.
        Returns a dict with tokens, targets, positions (or None), the prompt length and the
        answer tokens, the last two only meaningful when all prompts share a length."""
        cfg = self.cfg
        B, D = a.shape
        dev = a.device
        total, _ = add_digits(a, b)
        # One spare zero column so operands can be shown at the answer's width.
        a, b = F.pad(a, (0, 1)), F.pad(b, (0, 1))
        n = torch.maximum(la, lb)
        if cfg.blanks:
            # Aligned blankspace zero-pads the operands to the answer's width as well, so all
            # three numbers share one layout of digits and blanks.
            na = nb = nc = n + 1
        elif cfg.zero_pad:
            na = nb = n
            nc = n + 1
        else:
            na, nb, nc = la, lb, digit_count(total)

        if cfg.blanks:
            W = cfg.blanks
            width = n + 1
            if cfg.blanks_fixed:
                p = torch.full_like(n, W)
            elif train:
                p = width + (torch.rand(B, device=dev, generator=self.gen) * (W - width + 1)).long()
            else:
                p = width
            k = p - width
            slot = torch.arange(W, device=dev)
            if train:
                scores = torch.rand(B, W, device=dev, generator=self.gen)
                scores[slot >= p[:, None]] = 2.0
                blank = scores.argsort(1).argsort(1) < k[:, None]
            else:
                blank = (slot >= width[:, None]) & (slot < p[:, None])
        else:
            W = D + 1
            k = torch.zeros(B, dtype=torch.long, device=dev)
            blank = torch.zeros(B, W, dtype=torch.bool, device=dev)
        blanks_before = blank.cumsum(1) - blank.long()

        sa, sb, sc = na + k, nb + k, nc + k
        start_b = 2 + sa
        start_c = 3 + sa + sb
        end = start_c + sc
        L = int(end.max()) + 1
        t = torch.arange(L, device=dev).expand(B, L)

        j_a = t - 1
        j_b = t - start_b[:, None]
        j_c = t - start_c[:, None]
        in_a = (j_a >= 0) & (j_a < sa[:, None])
        in_b = (j_b >= 0) & (j_b < sb[:, None])
        in_c = (j_c >= 0) & (j_c < sc[:, None])
        j = torch.where(in_a, j_a, torch.where(in_b, j_b, j_c)).clamp(0, W - 1)
        is_blank = (in_a | in_b | in_c) & blank.gather(1, j)
        d = j - blanks_before.gather(1, j)

        def digits_of(num, length):
            pos = d if cfg.reverse else (length[:, None] - 1 - d)
            return num.gather(1, pos.clamp(0, num.shape[1] - 1))

        tokens = torch.full((B, L), PAD, dtype=torch.long, device=dev)
        tokens[:, 0] = END
        tokens = torch.where(in_a, digits_of(a, na), tokens)
        tokens = torch.where(in_b, digits_of(b, nb), tokens)
        tokens = torch.where(in_c, digits_of(total, nc), tokens)
        tokens[is_blank] = BLANK
        tokens[t == (1 + sa)[:, None]] = PLUS
        tokens[t == (2 + sa + sb)[:, None]] = EQ
        tokens[t == end[:, None]] = END

        answer = in_c | (t == end[:, None])
        targets = torch.full_like(tokens, -100)
        targets[:, :-1] = torch.where(answer[:, 1:], tokens[:, 1:], -100)

        positions = None
        if cfg.positions != "sequential":
            if train:
                offset = torch.randint(0, cfg.offset_max + 1, (B, 1), device=dev, generator=self.gen)
            else:
                offset = torch.zeros(B, 1, dtype=torch.long, device=dev)
            positions = torch.zeros(B, L, dtype=torch.long, device=dev)
            digit_slot = (in_a | in_b | in_c) & ~is_blank
            positions[digit_slot] = (offset + d + 1)[digit_slot]
            if cfg.positions == "abacus":
                # The end token takes the id the next answer digit would have had, so that
                # greedy decoding can assign ids without knowing the answer length.
                at_end = t == end[:, None]
                positions[at_end] = (offset + nc[:, None] + 1).expand(B, L)[at_end]
            elif cfg.positions == "coupled":
                ops = (t == (1 + sa)[:, None]) | (t == (2 + sa + sb)[:, None])
                positions[ops] = (offset + n[:, None] + 1).expand(B, L)[ops]

        prompt_len = int(start_c[0]) if bool((start_c == start_c[0]).all()) else None
        return {
            "tokens": tokens,
            "targets": targets,
            "positions": positions,
            "prompt_len": prompt_len,
            "answer": torch.where(answer, tokens, PAD) if prompt_len else None,
        }

    def train_batch(self, batch_size: int):
        a, b, la, lb = self.sample(batch_size, self.cfg.max_digits)
        ex = self.build(a, b, la, lb, train=True)
        return ex["tokens"], ex["targets"], ex["positions"]

    @torch.no_grad()
    def accuracy(self, model, ex, chunk: int = 64) -> float:
        p = ex["prompt_len"]
        n_new = ex["tokens"].shape[1] - p
        out = torch.cat([
            model.generate(ex["tokens"][i : i + chunk, :p], n_new,
                           None if ex["positions"] is None else ex["positions"][i : i + chunk])
            for i in range(0, len(ex["tokens"]), chunk)
        ])[:, p:]
        expected = ex["answer"][:, p:]
        if self.cfg.blanks:
            # Blanks in the answer are stripped before comparing, as in the paper.
            out, expected = strip_blanks(out), strip_blanks(expected)
        ok = (out == expected) | (expected == PAD) | (expected == BLANK)
        return ok.all(1).float().mean().item()

    def evaluate(self, model) -> dict:
        return {f"acc_{n}": self.accuracy(model, ex) for n, ex in self.test_sets.items()}
