from importlib.metadata import version

from rustling import taggers  # noqa: F401
from rustling import wordseg  # noqa: F401

__version__ = version("rustling")

__all__ = ["__version__", "taggers", "wordseg"]
