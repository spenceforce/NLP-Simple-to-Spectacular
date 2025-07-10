from typing import List


class CharTokenizer:
    """Encode and decode text."""

    def __init__(self, cls_token="<cls>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>"):
        self.cls_token = cls_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

    def train(self, X: List[str]):
        """Create a vocabulary from `X`."""
        vocabulary = set()
        for x in X:
            vocabulary |= set(x)
        self.tokens = list(vocabulary)
        self.special_tokens = [
            self.unk_token,
            self.cls_token,
            self.eos_token,
            self.pad_token,
        ]
        self.tokens.extend(self.special_tokens)
        self.tok2idx = {tok: i for i, tok in enumerate(self.tokens)}
        self.unk_idx = self.tok2idx[self.unk_token]
        self.cls_idx = self.tok2idx[self.cls_token]
        self.eos_idx = self.tok2idx[self.eos_token]
        self.pad_idx = self.tok2idx[self.pad_token]
        return self

    def tokenize(self, x: str) -> List[str]:
        """Tokenize `x`."""
        return [
            self.cls_token,
            *[tok if tok in self.tok2idx else self.unk_token for tok in x],
            self.eos_token,
        ]

    def encode(self, x: str) -> List[int]:
        """Encode `x`."""
        return [self.tok2idx[tok] for tok in self.tokenize(x)]

    def encode_batch(self, X: List[str]) -> List[List[int]]:
        """Encode each `str` in `X`."""
        rv = []
        for x in X:
            rv.append(self.encode(x))
        return rv

    def decode(self, x: List[int]) -> str:
        """Decode `x`."""
        return "".join([self.tokens[i] for i in x[1:-1]])

    def decode_batch(self, X: List[List[int]]) -> List[str]:
        """Decode each encoding in `X` to a `str`."""
        rv = []
        for x in X:
            rv.append(self.decode(x))
        return rv
