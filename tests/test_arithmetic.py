import pytest
import torch

from carrybit.arithmetic import Arithmetic, add_digits
from carrybit.config import ArithmeticConfig
from carrybit.tokenizer import BLANK, PAD, decode, encode

RUNGS = {
    "plain": dict(reverse=False),
    "reversed": dict(),
    "zeropad": dict(zero_pad=True),
    "abacus": dict(positions="abacus"),
    "coupled": dict(zero_pad=True, positions="coupled"),
    "blankspace": dict(zero_pad=True, blanks=8),
}


def make(**kw):
    cfg = ArithmeticConfig(max_digits=6, test_digits=(3, 5), test_examples=16, **kw)
    return Arithmetic(cfg, seed=0, device="cpu")


def parse(text: str, reverse: bool):
    """Turn '$a+b=c$' back into ints, ignoring blanks."""
    body = text.strip("$ ").replace("_", "")
    lhs, c = body.split("=")
    a, b = lhs.split("+")
    if reverse:
        a, b, c = a[::-1], b[::-1], c[::-1]
    return int(a), int(b), int(c)


def test_tokenizer_roundtrip():
    assert decode(encode("$12+340=352$")) == "$12+340=352$"


def test_add_digits_matches_python():
    a = torch.tensor([[9, 9, 9], [5, 0, 0]])
    b = torch.tensor([[1, 0, 0], [5, 0, 0]])
    total, carries = add_digits(a, b)
    assert total.tolist() == [[0, 0, 0, 1], [0, 1, 0, 0]]
    assert carries.tolist() == [[0, 1, 1], [0, 1, 0]]


@pytest.mark.parametrize("rung", RUNGS)
@pytest.mark.parametrize("train", [True, False])
def test_examples_are_correct_sums(rung, train):
    task = make(**RUNGS[rung])
    a, b, la, lb = task.sample(64, 6)
    ex = task.build(a, b, la, lb, train=train)
    for row in ex["tokens"]:
        x, y, z = parse(decode(row), task.cfg.reverse)
        assert x + y == z


@pytest.mark.parametrize("rung", RUNGS)
def test_targets_cover_answer_and_end_only(rung):
    task = make(**RUNGS[rung])
    ex = task.build(*task.sample(32, 6), train=True)
    tokens, targets = ex["tokens"], ex["targets"]
    for row_t, row_y in zip(tokens, targets):
        text = decode(row_t)
        eq = text.index("=")
        end = text.index("$", 1)
        supervised = (row_y != -100).nonzero().flatten().tolist()
        assert supervised == list(range(eq, end))
        assert row_y[eq:end].tolist() == row_t[eq + 1 : end + 1].tolist()


def test_sample_respects_lengths_and_leading_digits():
    task = make()
    a, b, la, lb = task.sample(500, 6, min_digits=1)
    for digits, length in ((a, la), (b, lb)):
        idx = torch.arange(6)
        assert (digits[idx >= length[:, None]] == 0).all()
        lead = digits.gather(1, (length - 1)[:, None]).flatten()
        assert (lead[length > 1] != 0).all()
    assert la.min() >= 1 and la.max() == 6


def test_zero_pad_makes_prompts_uniform_within_a_length():
    task = make(zero_pad=True)
    ex = task.test_sets[5]
    assert ex["prompt_len"] == 1 + 5 + 1 + 5 + 1
    assert (ex["tokens"][:, ex["prompt_len"] - 1] == encode("=")[0]).all()


def test_coupled_positions_share_ids_across_significance():
    task = make(zero_pad=True, positions="coupled")
    ex = task.build(*task.sample(8, 6), train=True)
    for tok, pos in zip(ex["tokens"], ex["positions"]):
        text = decode(tok)
        plus, eq, end = text.index("+"), text.index("="), text.index("$", 1)
        n = plus - 1
        a_ids, b_ids, c_ids = pos[1:plus], pos[plus + 1 : eq], pos[eq + 1 : end]
        assert torch.equal(a_ids, b_ids)
        assert torch.equal(c_ids[:n], a_ids)
        assert a_ids.tolist() == list(range(a_ids[0], a_ids[0] + n))
        assert pos[plus] == pos[eq] == c_ids[n]
        assert pos[0] == 0 and pos[end] == 0


def test_abacus_positions_restart_per_number():
    task = make(positions="abacus")
    ex = task.build(*task.sample(8, 6), train=False)
    for tok, pos in zip(ex["tokens"], ex["positions"]):
        text = decode(tok)
        plus, eq, end = text.index("+"), text.index("="), text.index("$", 1)
        for lo, hi in ((1, plus), (plus + 1, eq), (eq + 1, end)):
            assert pos[lo:hi].tolist() == list(range(1, hi - lo + 1))
        assert pos[end] == end - eq
        assert pos[plus] == pos[eq] == pos[0] == 0


def test_blanks_are_aligned_across_all_three_numbers():
    task = make(zero_pad=True, blanks=8)
    ex = task.build(*task.sample(32, 6), train=True)
    saw_blank = False
    for tok in ex["tokens"]:
        text = decode(tok)
        a, rest = text[1:].split("+")
        b, rest = rest.split("=")
        c = rest[: rest.index("$")]
        blank_idx = lambda s: [i for i, ch in enumerate(s) if ch == "_"]
        assert blank_idx(a) == blank_idx(b) == blank_idx(c)
        assert len(c) == len(a) + 1 and c[-1] != "_"
        saw_blank |= bool(blank_idx(a))
    assert saw_blank
    test = task.test_sets[3]["tokens"]
    assert (test != BLANK).all()


def test_accuracy_is_one_for_an_oracle():
    task = make(zero_pad=True)
    ex = task.test_sets[3]

    class Oracle:
        def generate(self, prompt, n_new, positions):
            return ex["tokens"][:, : prompt.shape[1] + n_new]

    assert task.accuracy(Oracle(), ex) == 1.0
