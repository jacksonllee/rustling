"""Language models.

This module provides n-gram language models that can be trained on
tokenized text and used to score and generate word sequences.
"""

from rustling._lib_name import lm as _lm

LanguageModel = _lm.LanguageModel


class MLE(LanguageModel):
    """Maximum Likelihood Estimation language model.

    An n-gram language model with no smoothing.
    """

    def __new__(cls, *, order: int):
        return super().__new__(cls, order=order, smoothing="mle")


class Lidstone(LanguageModel):
    """Lidstone (additive) smoothing language model.

    An n-gram language model with Lidstone smoothing, which adds
    a constant gamma to all counts.
    """

    def __new__(cls, *, order: int, gamma: float):
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0: {gamma}")
        return super().__new__(cls, order=order, smoothing="lidstone", gamma=gamma)


class Laplace(LanguageModel):
    """Laplace (add-one) smoothing language model.

    An n-gram language model with Laplace smoothing (Lidstone with gamma=1).
    """

    def __new__(cls, *, order: int):
        return super().__new__(cls, order=order, smoothing="laplace")


__all__ = ["LanguageModel", "MLE", "Lidstone", "Laplace"]
