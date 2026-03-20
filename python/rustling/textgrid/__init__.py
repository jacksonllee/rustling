"""TextGrid (Praat) file parsing.

This module provides a parser for Praat TextGrid annotation files
and data structures for accessing tiers, intervals, and points.
"""

from rustling._lib_name import textgrid as _textgrid
from rustling.textgrid._read_textgrid import read_textgrid

Interval = _textgrid.Interval
IntervalTier = _textgrid.IntervalTier
Point = _textgrid.Point
TextGrid = _textgrid.TextGrid
TextTier = _textgrid.TextTier

__all__ = [
    "Interval",
    "IntervalTier",
    "Point",
    "TextGrid",
    "TextTier",
    "read_textgrid",
]
