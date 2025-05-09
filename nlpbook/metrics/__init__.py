"""Machine learning metrics for evaluating the performance of models."""

import numpy as np


def perplexity(probabilities):
    """Calculate the perplexity of a sequence of probabilities.

    Parameters
    ----------
    probabilities : list of float
        List of probabilities.

    Returns
    -------
    float
        Perplexity of the sequence.
    """
    return np.exp(-np.sum(np.log(probabilities)) / len(probabilities))
