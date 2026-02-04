"""Type stubs for rustling.wordseg."""

from __future__ import annotations

from typing import Sequence

class LongestStringMatching:
    """Longest string matching segmenter.

    This model constructs predicted words by moving from left to right
    along an unsegmented sentence and finding the longest matching words,
    constrained by a maximum word length parameter.
    """

    def __init__(self, *, max_word_length: int) -> None:
        """Initialize a longest string matching segmenter.

        Parameters
        ----------
        max_word_length : int
            Maximum word length in the segmented sentences during prediction.
            Must be >= 2 to be meaningful.

        Raises
        ------
        ValueError
            If max_word_length is < 2.
        """
        ...

    def fit(self, sents: Sequence[Sequence[str]]) -> None:
        """Train the model with the input segmented sentences.

        No cleaning or preprocessing (e.g., normalizing upper/lowercase,
        tokenization) is performed on the training data.

        Parameters
        ----------
        sents : Sequence[Sequence[str]]
            An iterable of segmented sentences (each sentence is a
            sequence of words).
        """
        ...

    def predict(self, sent_strs: Sequence[str]) -> list[list[str]]:
        """Segment the given unsegmented sentences.

        Parameters
        ----------
        sent_strs : Sequence[str]
            An iterable of unsegmented sentences.

        Returns
        -------
        list[list[str]]
            A list of segmented sentences.
        """
        ...

class RandomSegmenter:
    """A random segmenter.

    Segmentation is predicted at random at each potential word
    boundary independently for a given probability. No training is required.
    """

    def __init__(self, *, prob: float) -> None:
        """Initialize a random segmenter.

        Parameters
        ----------
        prob : float
            The probability from [0, 1) that segmentation occurs
            between two symbols.

        Raises
        ------
        ValueError
            If prob is outside [0, 1).
        """
        ...

    def fit(self, sents: Sequence[Sequence[str]]) -> None:
        """Training is not required for RandomSegmenter.

        Parameters
        ----------
        sents : Sequence[Sequence[str]]
            Unused.

        Raises
        ------
        NotImplementedError
            Always, since no training is needed.
        """
        ...

    def predict(self, sent_strs: Sequence[str]) -> list[list[str]]:
        """Segment the given unsegmented sentences.

        Parameters
        ----------
        sent_strs : Sequence[str]
            An iterable of unsegmented sentences.

        Returns
        -------
        list[list[str]]
            A list of segmented sentences.
        """
        ...

__all__ = ["LongestStringMatching", "RandomSegmenter"]
