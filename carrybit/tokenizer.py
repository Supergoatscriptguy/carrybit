VOCAB = "0123456789+-=$_ "
PLUS, MINUS, EQ, END, BLANK, PAD = (VOCAB.index(c) for c in "+-=$_ ")


def encode(text: str) -> list[int]:
    return [VOCAB.index(c) for c in text]


def decode(ids) -> str:
    return "".join(VOCAB[int(i)] for i in ids).rstrip(" ")
