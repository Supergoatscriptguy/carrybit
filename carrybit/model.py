import torch
import torch.nn as nn
import torch.nn.functional as F

from carrybit.config import ModelConfig


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        # Set record=True to keep the last attention pattern around for inspection.
        self.record = False
        self.pattern = None

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.record:
            scores = q @ k.transpose(-1, -2) / k.shape[-1] ** 0.5
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            self.pattern = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            y = self.pattern @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
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
        self.pos_embed = nn.Embedding(cfg.max_positions, cfg.d_model) if cfg.positional else None
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
