"""Word segmentation."""

from __future__ import annotations

import os
from typing import Literal, Sequence, overload

from rustling.seq_feature import SeqFeatureTemplate

class DAGHMMSegmenter:
    """A DAG + HMM hybrid word segmenter (jieba-style).

    Layer 1: Dictionary-based DAG with backward dynamic programming.
    Layer 2: HMM fallback (BMES tagger) for out-of-vocabulary spans.
    """

    def __init__(
        self,
        *,
        n_iter: int | None = None,
        tolerance: float | None = None,
        gamma: float | None = None,
        random_seed: int | None = None,
        features: Sequence[SeqFeatureTemplate] | None = None,
    ) -> None:
        """Initialize a DAG + HMM hybrid segmenter.

        Args:
            n_iter: Maximum EM iterations for the HMM component's
                unsupervised fitting (default 1).
            tolerance: Convergence threshold for EM (default 0.0).
            gamma: Lidstone smoothing parameter for the HMM component
                (default 1.0).
            random_seed: Optional seed for reproducible random
                initialization.
            features: Optional list of observation feature templates
                created with ``seq_obs()``.
        """
        ...

    def fit_segmented(self, sents: Sequence[Sequence[str]]) -> None:
        """Train the model with supervised segmented sentences.

        Builds the dictionary from word frequencies and trains the HMM
        component on the same data.

        Args:
            sents: An iterable of segmented sentences (each sentence is
                a sequence of words).
        """
        ...

    def fit_unsegmented(self, sent_strs: Sequence[str]) -> None:
        """Refine the HMM component with unsupervised EM.

        Args:
            sent_strs: An iterable of unsegmented sentences.
        """
        ...

    def score(self, sents: Sequence[Sequence[str]]) -> list[float]:
        """Compute log-likelihood of segmented sentences under the model.

        Uses the Forward algorithm on the HMM component.

        Args:
            sents: Segmented sentences (each sentence is a sequence of words).

        Returns:
            Log-likelihood for each sentence.

        Raises:
            ValueError: If the model has not been fitted.
        """
        ...

    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: Literal[False] = False
    ) -> list[list[str]]: ...
    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: Literal[True]
    ) -> list[list[tuple[str, tuple[int, int]]]]: ...
    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: bool = False
    ) -> list[list[str]] | list[list[tuple[str, tuple[int, int]]]]: ...
    def predict(  # type: ignore[misc]
        self, sent_strs: Sequence[str], *, offsets: bool = False
    ) -> list[list[str]] | list[list[tuple[str, tuple[int, int]]]]:
        """Segment the given unsegmented sentences.

        Args:
            sent_strs: An iterable of unsegmented sentences.
            offsets: If True, return each word as a tuple of
                ``(word, (start, end))`` where start and end are
                character indices (exclusive end, like Python slices).

        Returns:
            A list of segmented sentences. When *offsets* is True,
            each word is a ``(word, (start, end))`` tuple.
        """
        ...

    def save(
        self,
        path: str | os.PathLike[str],
        metadata: dict[str, str],
    ) -> None:
        """Save the model and metadata to a zstd-compressed FlatBuffers binary.

        Args:
            path: The file path to save the model to.
                The file extension name ``.fb.zst`` is recommended.
            metadata: Arbitrary key-value metadata to store alongside
                the model (e.g., PUA character mappings).
        """
        ...

    def load(
        self,
        path: str | os.PathLike[str],
    ) -> dict[str, str]:
        """Load a model and metadata from a binary file.

        Args:
            path: The file path to load the model from.

        Returns:
            The metadata dictionary stored in the file.
        """
        ...

class HiddenMarkovModelSegmenter:
    """An HMM-based word segmenter using supervised BMES tagging.

    This model uses a Hidden Markov Model where the hidden states are
    BMES (Begin/Middle/End/Single) labels and the observations are
    characters. Training directly computes HMM parameters from supervised
    data. Decoding uses the Viterbi algorithm.
    """

    def __init__(
        self,
        *,
        n_iter: int = 1,
        tolerance: float = 0.0,
        gamma: float = 1.0,
        random_seed: int | None = None,
        features: Sequence[SeqFeatureTemplate] | None = None,
    ) -> None:
        """Initialize an HMM-based word segmenter.

        Args:
            n_iter: Maximum EM iterations for unsupervised fitting
                (default 1).
            tolerance: Convergence threshold for EM (default 0.0).
            gamma: Lidstone smoothing parameter for supervised
                training. Must be > 0. A value of 1.0 (default)
                corresponds to Laplace (add-one) smoothing.
            random_seed: Optional seed for reproducible random
                initialization.
            features: Optional list of observation feature templates
                created with ``seq_obs()``. Only observation features
                are supported (not ``seq_label()``). If ``None``, uses
                the following default features:
                ``seq_obs(-1)``, ``seq_obs(0)``, ``seq_obs(1)``,
                ``seq_obs(-1, 0)``, ``seq_obs(0, 1)``,
                ``seq_obs(-1, 1)``.

        Raises:
            ValueError: If ``gamma`` <= 0 or label features are
                provided.
        """
        ...

    def fit_segmented(self, sents: Sequence[Sequence[str]]) -> None:
        """Train the model with supervised segmented sentences.

        No cleaning or preprocessing (e.g., normalizing upper/lowercase,
        tokenization) is performed on the training data.

        Args:
            sents: An iterable of segmented sentences (each sentence is
                a sequence of words).
        """
        ...

    def fit_unsegmented(self, sent_strs: Sequence[str]) -> None:
        """Train the model with unsupervised unsegmented sentences.

        Uses the Baum-Welch (EM) algorithm. If the model was previously
        fitted (e.g., via ``fit_segmented``), the existing parameters
        serve as EM initialization (warm start).

        Args:
            sent_strs: An iterable of unsegmented sentences.
        """
        ...

    def score(self, sents: Sequence[Sequence[str]]) -> list[float]:
        """Compute log-likelihood of segmented sentences under the model.

        Uses the Forward algorithm on the underlying HMM.

        Args:
            sents: Segmented sentences (each sentence is a sequence of words).

        Returns:
            Log-likelihood for each sentence.

        Raises:
            ValueError: If the model has not been fitted.
        """
        ...

    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: Literal[False] = False
    ) -> list[list[str]]: ...
    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: Literal[True]
    ) -> list[list[tuple[str, tuple[int, int]]]]: ...
    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: bool = False
    ) -> list[list[str]] | list[list[tuple[str, tuple[int, int]]]]: ...
    def predict(  # type: ignore[misc]
        self, sent_strs: Sequence[str], *, offsets: bool = False
    ) -> list[list[str]] | list[list[tuple[str, tuple[int, int]]]]:
        """Segment the given unsegmented sentences.

        Args:
            sent_strs: An iterable of unsegmented sentences.
            offsets: If True, return each word as a tuple of
                ``(word, (start, end))`` where start and end are
                character indices (exclusive end, like Python slices).

        Returns:
            A list of segmented sentences. When *offsets* is True,
            each word is a ``(word, (start, end))`` tuple.
        """
        ...

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the model to a zstd-compressed FlatBuffers binary.

        Args:
            path: The path where the model will be saved.
                The file extension name ``.fb.zst`` is recommended.
        """
        ...

    def load(self, path: str | os.PathLike[str]) -> None:
        """Load a model.

        Args:
            path: The path where the model, stored as a zstd-compressed FlatBuffers
                binary, is located.
        """
        ...

class LongestStringMatching:
    """Longest string matching segmenter.

    This model constructs predicted words by moving from left to right
    along an unsegmented sentence and finding the longest matching words,
    constrained by a maximum word length parameter.
    """

    def __init__(self, *, max_word_length: int) -> None:
        """Initialize a longest string matching segmenter.

        Args:
            max_word_length: Maximum word length in the segmented
                sentences during prediction. Must be >= 2 to be
                meaningful.

        Raises:
            ValueError: If max_word_length is < 2.
        """
        ...

    def fit(self, sents: Sequence[Sequence[str]]) -> None:
        """Train the model with the input segmented sentences.

        No cleaning or preprocessing (e.g., normalizing upper/lowercase,
        tokenization) is performed on the training data.

        Args:
            sents: An iterable of segmented sentences (each sentence is
                a sequence of words).
        """
        ...

    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: Literal[False] = False
    ) -> list[list[str]]: ...
    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: Literal[True]
    ) -> list[list[tuple[str, tuple[int, int]]]]: ...
    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: bool = False
    ) -> list[list[str]] | list[list[tuple[str, tuple[int, int]]]]: ...
    def predict(  # type: ignore[misc]
        self, sent_strs: Sequence[str], *, offsets: bool = False
    ) -> list[list[str]] | list[list[tuple[str, tuple[int, int]]]]:
        """Segment the given unsegmented sentences.

        Args:
            sent_strs: An iterable of unsegmented sentences.
            offsets: If True, return each word as a tuple of
                ``(word, (start, end))`` where start and end are
                character indices (exclusive end, like Python slices).

        Returns:
            A list of segmented sentences. When *offsets* is True,
            each word is a ``(word, (start, end))`` tuple.
        """
        ...

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the model to a zstd-compressed FlatBuffers binary.

        Args:
            path: The path where the model will be saved.
                The file extension name ``.fb.zst`` is recommended.
        """
        ...

    def load(self, path: str | os.PathLike[str]) -> None:
        """Load a model.

        Args:
            path: The path where the model, stored as a zstd-compressed FlatBuffers
                binary, is located.
        """
        ...

class RandomSegmenter:
    """A random segmenter.

    Segmentation is predicted at random at each potential word
    boundary independently for a given probability. No training is required.
    """

    def __init__(self, *, prob: float) -> None:
        """Initialize a random segmenter.

        Args:
            prob: The probability from [0, 1) that segmentation occurs
                between two symbols.

        Raises:
            ValueError: If prob is outside [0, 1).
        """
        ...

    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: Literal[False] = False
    ) -> list[list[str]]: ...
    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: Literal[True]
    ) -> list[list[tuple[str, tuple[int, int]]]]: ...
    @overload
    def predict(
        self, sent_strs: Sequence[str], *, offsets: bool = False
    ) -> list[list[str]] | list[list[tuple[str, tuple[int, int]]]]: ...
    def predict(  # type: ignore[misc]
        self, sent_strs: Sequence[str], *, offsets: bool = False
    ) -> list[list[str]] | list[list[tuple[str, tuple[int, int]]]]:
        """Segment the given unsegmented sentences.

        Args:
            sent_strs: An iterable of unsegmented sentences.
            offsets: If True, return each word as a tuple of
                ``(word, (start, end))`` where start and end are
                character indices (exclusive end, like Python slices).

        Returns:
            A list of segmented sentences. When *offsets* is True,
            each word is a ``(word, (start, end))`` tuple.
        """
        ...

__all__ = [
    "DAGHMMSegmenter",
    "HiddenMarkovModelSegmenter",
    "LongestStringMatching",
    "RandomSegmenter",
]
