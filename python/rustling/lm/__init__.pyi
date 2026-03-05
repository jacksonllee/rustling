"""Language models."""

from __future__ import annotations

import os
from typing import Sequence

class MLE:
    """Maximum Likelihood Estimation language model.

    An n-gram language model with no smoothing.
    """

    def __init__(self, *, order: int) -> None:
        """Initialize an MLE language model.

        Args:
            order: The order of the n-gram model. Must be >= 1.

        Raises:
            ValueError: If order < 1.
        """
        ...

    def fit(self, sents: Sequence[Sequence[str]]) -> None:
        """Train the language model on tokenized sentences.

        Each sentence is a list of tokens. The model extracts n-grams of all
        orders from 1 to the model order and counts their occurrences.
        Sentences are automatically padded with ``<s>`` and ``</s>`` tokens.

        Args:
            sents: An iterable of tokenized sentences.
        """
        ...

    def score(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the probability of a word given a context.

        Maps out-of-vocabulary words to ``<UNK>`` via the vocabulary.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            The probability P(word | context).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def unmasked_score(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the probability of a word given a context, without OOV mapping.

        Unlike ``score``, this method does not map out-of-vocabulary words
        to ``<UNK>``.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            The probability P(word | context).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def logscore(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the log (base 2) probability of a word given a context.

        Maps out-of-vocabulary words to ``<UNK>`` via the vocabulary.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            log2(P(word | context)). Returns negative infinity if
            probability is 0.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def generate(
        self,
        *,
        num_words: int = 1,
        text_seed: Sequence[str] | None = None,
        random_seed: int | None = None,
    ) -> list[str]:
        """Generate words from the language model.

        Uses weighted random sampling from the conditional distribution.
        Generation stops early if ``</s>`` (end-of-sentence) is sampled
        or if no continuations are available for the current context.

        Args:
            num_words: Number of words to generate.
            text_seed: Seed text (context to start from). Defaults to
                beginning-of-sentence context.
            random_seed: Random seed for reproducibility.

        Returns:
            A list of generated words.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    @property
    def order(self) -> int:
        """The order of the n-gram model."""
        ...

    @property
    def vocab_size(self) -> int:
        """The vocabulary size (including special tokens)."""
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

        Raises:
            FileNotFoundError: If the file does not exist.
            EnvironmentError: If the file cannot be read as a language model
                or the smoothing/order does not match.
        """
        ...

class Lidstone:
    """Lidstone (additive) smoothing language model.

    An n-gram language model with Lidstone smoothing, which adds
    a constant gamma to all counts.
    """

    def __init__(self, *, order: int, gamma: float) -> None:
        """Initialize a Lidstone language model.

        Args:
            order: The order of the n-gram model. Must be >= 1.
            gamma: The smoothing parameter. Must be > 0.

        Raises:
            ValueError: If order < 1 or gamma <= 0.
        """
        ...

    def fit(self, sents: Sequence[Sequence[str]]) -> None:
        """Train the language model on tokenized sentences.

        Each sentence is a list of tokens. The model extracts n-grams of all
        orders from 1 to the model order and counts their occurrences.
        Sentences are automatically padded with ``<s>`` and ``</s>`` tokens.

        Args:
            sents: An iterable of tokenized sentences.
        """
        ...

    def score(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the probability of a word given a context.

        Maps out-of-vocabulary words to ``<UNK>`` via the vocabulary.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            The probability P(word | context).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def unmasked_score(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the probability of a word given a context, without OOV mapping.

        Unlike ``score``, this method does not map out-of-vocabulary words
        to ``<UNK>``.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            The probability P(word | context).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def logscore(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the log (base 2) probability of a word given a context.

        Maps out-of-vocabulary words to ``<UNK>`` via the vocabulary.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            log2(P(word | context)). Returns negative infinity if
            probability is 0.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def generate(
        self,
        *,
        num_words: int = 1,
        text_seed: Sequence[str] | None = None,
        random_seed: int | None = None,
    ) -> list[str]:
        """Generate words from the language model.

        Uses weighted random sampling from the conditional distribution.
        Generation stops early if ``</s>`` (end-of-sentence) is sampled
        or if no continuations are available for the current context.

        Args:
            num_words: Number of words to generate.
            text_seed: Seed text (context to start from). Defaults to
                beginning-of-sentence context.
            random_seed: Random seed for reproducibility.

        Returns:
            A list of generated words.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    @property
    def order(self) -> int:
        """The order of the n-gram model."""
        ...

    @property
    def vocab_size(self) -> int:
        """The vocabulary size (including special tokens)."""
        ...

    @property
    def gamma(self) -> float:
        """The smoothing parameter."""
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

        Raises:
            FileNotFoundError: If the file does not exist.
            EnvironmentError: If the file cannot be read as a language model
                or the smoothing/order/gamma does not match.
        """
        ...

class Laplace:
    """Laplace (add-one) smoothing language model.

    An n-gram language model with Laplace smoothing (Lidstone with gamma=1).
    """

    def __init__(self, *, order: int) -> None:
        """Initialize a Laplace language model.

        Args:
            order: The order of the n-gram model. Must be >= 1.

        Raises:
            ValueError: If order < 1.
        """
        ...

    def fit(self, sents: Sequence[Sequence[str]]) -> None:
        """Train the language model on tokenized sentences.

        Each sentence is a list of tokens. The model extracts n-grams of all
        orders from 1 to the model order and counts their occurrences.
        Sentences are automatically padded with ``<s>`` and ``</s>`` tokens.

        Args:
            sents: An iterable of tokenized sentences.
        """
        ...

    def score(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the probability of a word given a context.

        Maps out-of-vocabulary words to ``<UNK>`` via the vocabulary.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            The probability P(word | context).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def unmasked_score(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the probability of a word given a context, without OOV mapping.

        Unlike ``score``, this method does not map out-of-vocabulary words
        to ``<UNK>``.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            The probability P(word | context).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def logscore(self, word: str, context: Sequence[str] | None = None) -> float:
        """Return the log (base 2) probability of a word given a context.

        Maps out-of-vocabulary words to ``<UNK>`` via the vocabulary.

        Args:
            word: The word to score.
            context: The preceding context words.

        Returns:
            log2(P(word | context)). Returns negative infinity if
            probability is 0.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def generate(
        self,
        *,
        num_words: int = 1,
        text_seed: Sequence[str] | None = None,
        random_seed: int | None = None,
    ) -> list[str]:
        """Generate words from the language model.

        Uses weighted random sampling from the conditional distribution.
        Generation stops early if ``</s>`` (end-of-sentence) is sampled
        or if no continuations are available for the current context.

        Args:
            num_words: Number of words to generate.
            text_seed: Seed text (context to start from). Defaults to
                beginning-of-sentence context.
            random_seed: Random seed for reproducibility.

        Returns:
            A list of generated words.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    @property
    def order(self) -> int:
        """The order of the n-gram model."""
        ...

    @property
    def vocab_size(self) -> int:
        """The vocabulary size (including special tokens)."""
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

        Raises:
            FileNotFoundError: If the file does not exist.
            EnvironmentError: If the file cannot be read as a language model
                or the smoothing/order does not match.
        """
        ...

__all__ = ["MLE", "Lidstone", "Laplace"]
