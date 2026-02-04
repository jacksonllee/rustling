"""Word segmentation models.

This module provides word segmentation models that can be trained on
segmented sentences and used to predict segmentation of unsegmented text.
"""

from rustling._lib_name import wordseg as _wordseg

LongestStringMatching = _wordseg.LongestStringMatching
RandomSegmenter = _wordseg.RandomSegmenter

__all__ = ["LongestStringMatching", "RandomSegmenter"]
