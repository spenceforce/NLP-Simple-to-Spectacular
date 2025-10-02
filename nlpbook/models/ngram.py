"""N-gram models for text generation."""
import numpy as np


class Unigram:
    """Unigram model for text generation."""

    def __init__(self, tokenizer, seed=None):
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)

    def fit(self, X):
        """Fit the model to the data.

        Parameters
        ----------
        X : list of list of int
            List of encodings to fit the model to.

        Returns
        -------
        self : Unigram
            Fitted model.
        """
        # Start with a count of 1 for every token.
        encoding_counts = np.ones(len(self.tokenizer.tokens))
        for encoding in X:
            # Get the encoding values and their counts.
            unique, counts = np.unique(encoding, return_counts=True)
            # Add each count to it's respective index.
            encoding_counts[unique] += counts
        self.counts_ = encoding_counts
        # Convert the counts to frequencies.
        self.probabilities_ = encoding_counts / encoding_counts.sum()

        return self

    def _sample(self):
        """Sample a single encoding.

        Returns
        -------
        encoding : list of int
            Sampled encoding.
        """
        values = list(range(len(self.tokenizer.tokens)))
        encoding = [self.tokenizer.cls_idx]
        while encoding[-1] != self.tokenizer.eos_idx:
            encoding.append(self.rng.choice(values, p=self.probabilities_))
        return encoding

    def sample(self, n=1):
        """Sample `n` encodings.

        Parameters
        ----------
        n : int
            Number of encodings to sample.

        Returns
        -------
        encodings : list of list of int or list of int
            Sampled encodings.
        """
        assert n > 0, "Cannot generate a nonpositive number of samples."
        if n == 1:
            return self._sample()
        return [self._sample() for _ in range(n)]

    def probabilities(self, encoding):
        """Get the probabilities of the tokens in the encoding.
        Parameters
        ----------
        encoding : list of int
            Encoding to get the probabilities for.

        Returns
        -------
        probabilities : list of float
            Probabilities of the tokens in the encoding.
        """
        return np.array([self.probabilities_[x] for x in encoding])
