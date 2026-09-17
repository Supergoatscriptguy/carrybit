import torch
import torch.nn as nn
import torch.nn.functional as F

from carrybit.config import ModelConfig


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout
        # Shaw-style relative positions as a learned per-head bias on the scores, indexed by
        # how far back the key is. Causal attention only ever looks back, so one table suffices.
        self.rel_bias = nn.Embedding(cfg.max_positions, cfg.n_heads) if cfg.pos_embed == "relative" else None
        # Set record=True to keep the last attention pattern around for inspection.
        self.record = False
        self.pattern = None

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        bias = None
        if self.rel_bias is not None:
            idx = torch.arange(T, device=x.device)
            dist = (idx[:, None] - idx[None, :]).clamp(0, self.rel_bias.num_embeddings - 1)
            bias = self.rel_bias(dist).permute(2, 0, 1)
        if self.record:
            scores = q @ k.transpose(-1, -2) / k.shape[-1] ** 0.5
            if bias is not None:
                scores = scores + bias
            self.pattern = scores.masked_fill(~causal, float("-inf")).softmax(-1)
            y = self.pattern @ v
        elif bias is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            mask = bias.masked_fill(~causal, float("-inf")).to(q.dtype)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.ReLU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Decoder-only transformer. Position ids can be passed explicitly so a task can
    implement Abacus or coupled positions without the model knowing about them."""

    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_positions, cfg.d_model) if cfg.pos_embed == "absolute" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, positions=None):
        x = self.embed(tokens)
        if self.pos_embed is not None:
            if positions is None:
                positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = x + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.ln_f(x))

    @torch.no_grad()
    def generate(self, tokens, n_new: int, positions=None):
        """Greedy decoding. If positions are given they must cover the full final length."""
        for _ in range(n_new):
            pos = None if positions is None else positions[:, : tokens.shape[1]]
            logits = self(tokens, pos)[:, -1]
            tokens = torch.cat([tokens, logits.argmax(-1, keepdim=True)], dim=1)
        return tokens

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TwoHotMLP(nn.Module):
    """The one-hidden-layer ReLU MLP from Swaroop (2026): the two operands enter as a
    concatenated pair of one-hot vectors. Takes the same [a, b, =] tokens as the transformer
    and returns logits at every position so the training loop needs no special case."""

    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = nn.Linear(2 * vocab_size, cfg.d_mlp)
        self.out = nn.Linear(cfg.d_mlp, vocab_size)

    def forward(self, tokens, positions=None):
        x = torch.cat([F.one_hot(tokens[:, 0], self.vocab_size), F.one_hot(tokens[:, 1], self.vocab_size)], 1)
        logits = self.out(F.relu(self.hidden(x.float())))
        return logits[:, None].expand(-1, tokens.shape[1], -1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(vocab_size: int, cfg: ModelConfig):
    if cfg.arch == "mlp":
        return TwoHotMLP(vocab_size, cfg)
    return Transformer(vocab_size, cfg)
