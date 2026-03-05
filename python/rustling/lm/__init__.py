"""Language models.

This module provides n-gram language models that can be trained on
tokenized text and used to score and generate word sequences.
"""

from rustling._lib_name import lm as _lm

MLE = _lm.MLE
Lidstone = _lm.Lidstone
Laplace = _lm.Laplace

__all__ = ["MLE", "Lidstone", "Laplace"]
