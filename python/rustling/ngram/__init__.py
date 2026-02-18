"""N-gram counting.

This module provides an n-gram counter for counting n-gram
frequencies from sequential data.
"""

from rustling._lib_name import ngram as _ngram

Ngrams = _ngram.Ngrams

__all__ = ["Ngrams"]
