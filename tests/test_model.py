import torch

from carrybit.config import ModelConfig
from carrybit.model import Transformer


def test_relative_attention_is_causal_and_position_aware():
    cfg = ModelConfig(d_model=32, n_layers=1, n_heads=2, d_mlp=64, max_positions=16, pos_embed="relative")
    model = Transformer(10, cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        x = torch.randint(0, 10, (1, 8))
        y = torch.randint(0, 10, (1, 8))
        y[:, :5] = x[:, :5]
        assert torch.allclose(model(x)[:, :5], model(y)[:, :5], atol=1e-5)
        shifted = torch.cat([x[:, 1:], x[:, :1]], dim=1)
        assert not torch.allclose(model(x)[:, -1], model(shifted)[:, -1], atol=1e-3)
