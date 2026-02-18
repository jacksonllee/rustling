from importlib.metadata import version

from rustling import chat  # noqa: F401
from rustling import lm  # noqa: F401
from rustling import ngram  # noqa: F401
from rustling import tagging  # noqa: F401
from rustling import wordseg  # noqa: F401

__version__ = version("rustling")

__all__ = ["__version__", "chat", "lm", "ngram", "tagging", "wordseg"]
