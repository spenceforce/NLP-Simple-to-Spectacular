class CharTokenizer:
    """Encode and decode text."""

    def fit(self, X):
        """Create a vocabulary from `X`."""
        vocabulary = set()
        for x in X:
            vocabulary |= set(x)
        self.tokens = list(vocabulary)
        self.unk_token = "<unk>"
        self.cls_token = "<cls>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.unk_token, self.cls_token, self.eos_token]
        self.tokens.extend(self.special_tokens)
        self.tok2idx = {tok: i for i, tok in enumerate(self.tokens)}
        self.unk_idx = self.tok2idx[self.unk_token]
        self.cls_idx = self.tok2idx[self.cls_token]
        self.eos_idx = self.tok2idx[self.eos_token]
        return self

    def tokenize(self, x):
        """Tokenize `x`."""
        return [self.cls_token, *[tok if tok in self.tok2idx else self.unk_token for tok in x], self.eos_token]

    def encode(self, X):
        """Encode each `str` in `X`."""
        rv = []
        for x in X:
            rv.append(
                [self.tok2idx[tok] for tok in self.tokenize(x)]
            )
        return rv

    def decode(self, X):
        """Decode each encoding in `X` to a `str`."""
        rv = []
        for x in X:
            self.verify_encoding(x)
            rv.append("".join([self.tokens[i] for i in x[1:-1]]))
        return rv

    def verify_encoding(self, x):
        """Verify the encoding of `x`."""
        # Check the start and end tokens are present.
        assert x[0] == self.cls_idx
        assert x[-1] == self.eos_idx
