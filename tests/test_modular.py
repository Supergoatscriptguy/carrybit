import torch

from carrybit.config import ModularConfig
from carrybit.modular import ModularAddition


def test_split_covers_every_pair_once():
    task = ModularAddition(ModularConfig(p=23, train_frac=0.3), seed=0, device="cpu")
    train, test = task.train[0], task.test[0]
    assert len(train) + len(test) == 23 * 23
    seen = set(map(tuple, torch.cat([train, test])[:, :2].tolist()))
    assert len(seen) == 23 * 23


def test_labels_and_split_are_seed_stable():
    a = ModularAddition(ModularConfig(p=23), seed=1, device="cpu")
    b = ModularAddition(ModularConfig(p=23), seed=1, device="cpu")
    tokens, targets = a.train
    assert torch.equal(tokens, b.train[0])
    assert torch.equal(targets[:, -1], (tokens[:, 0] + tokens[:, 1]) % 23)
    assert (targets[:, :-1] == -100).all()
