"""Common utilities for Rustling benchmarks."""

from .data import hmm_data, lm_data, load_hkcancor, tagging_data, wordseg_data

__all__ = [
    "hmm_data",
    "lm_data",
    "load_hkcancor",
    "tagging_data",
    "wordseg_data",
]
