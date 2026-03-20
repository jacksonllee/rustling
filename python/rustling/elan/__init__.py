"""ELAN (.eaf) file parsing.

This module provides a parser for ELAN annotation files
and data structures for accessing tiers and annotations.
"""

from rustling._lib_name import elan as _elan
from rustling.elan._read_elan import read_elan

Annotation = _elan.Annotation
ELAN = _elan.ELAN
Tier = _elan.Tier

__all__ = [
    "Annotation",
    "ELAN",
    "Tier",
    "read_elan",
]
