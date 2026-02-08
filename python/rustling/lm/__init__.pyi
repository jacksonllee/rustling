"""Type stubs for rustling.lm."""

from __future__ import annotations

from typing import Sequence

class LanguageModel:
    """An n-gram language model.

    Supports MLE, Lidstone, and Laplace smoothing methods.
    """

    def __init__(
        self,
        *,
        order: int,
        smoothing: str = "mle",
        gamma: float = 1.0,
    ) -> None:
        """Initialize a language model.

        Args:
            order: The order of the n-gram model (e.g., 2 for bigram).
                Must be >= 1.
            smoothing: The smoothing method. One of "mle", "lidstone",
                "laplace".
            gamma: The smoothing parameter for Lidstone smoothing.
                Must be > 0. Only used when smoothing is "lidstone".

        Raises:
            ValueError: If order < 1, or smoothing is unknown, or
                gamma <= 0.
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

class MLE(LanguageModel):
    """Maximum Likelihood Estimation language model.

    An n-gram language model with no smoothing.
    """

    def __init__(self, *, order: int) -> None:
        """Initialize an MLE language model.

        Args:
            order: The order of the n-gram model. Must be >= 1.
        """
        ...

class Lidstone(LanguageModel):
    """Lidstone (additive) smoothing language model.

    An n-gram language model with Lidstone smoothing, which adds
    a constant gamma to all counts.
    """

    def __init__(self, *, order: int, gamma: float) -> None:
        """Initialize a Lidstone language model.

        Args:
            order: The order of the n-gram model. Must be >= 1.
            gamma: The smoothing parameter. Must be > 0.
        """
        ...

class Laplace(LanguageModel):
    """Laplace (add-one) smoothing language model.

    An n-gram language model with Laplace smoothing (Lidstone with gamma=1).
    """

    def __init__(self, *, order: int) -> None:
        """Initialize a Laplace language model.

        Args:
            order: The order of the n-gram model. Must be >= 1.
        """
        ...

__all__ = ["LanguageModel", "MLE", "Lidstone", "Laplace"]
