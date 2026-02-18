"""Tools for keeping track of and counting ngrams."""

from __future__ import annotations

from collections import Counter
from typing import Iterator, Sequence

class Ngrams:
    """An counter for storing n-grams efficiently and counting their frequencies.

    Accumulates n-gram counts from sequences of elements. N-grams
    do not cross sequence boundaries.
    """

    def __init__(self, n: int, *, min_n: int | None = None) -> None:
        """Create a new empty Ngrams.

        Args:
            n: The n-gram order (1 for unigrams, 2 for bigrams, etc.).
                Must be >= 1.
            min_n: The minimum n-gram order. If None, defaults to n
                (only n-grams of exactly order n). Must be >= 1 and <= n.

        Raises:
            ValueError: If n < 1, min_n < 1, or min_n > n.
        """
        ...

    def count(self, seq: Sequence[str]) -> None:
        """Count n-grams from a single sequence.

        Args:
            seq: A sequence of elements to extract n-grams from.
        """
        ...

    def count_seqs(self, seqs: Sequence[Sequence[str]]) -> None:
        """Count n-grams from multiple sequences.

        Args:
            seqs: An iterable of sequences.
        """
        ...

    def get(self, ngram: Sequence[str]) -> int:
        """Return the count for a specific n-gram.

        Args:
            ngram: The n-gram to look up.

        Returns:
            The count, or 0 if not observed.
        """
        ...

    def most_common(
        self, n: int | None = None, *, order: int | None = None
    ) -> list[tuple[tuple[str, ...], int]]:
        """Return the n most common n-grams with their counts.

        Args:
            n: Number of top entries to return. If None, returns all
                n-grams sorted by count (descending).
            order: If specified, only return n-grams of this specific order.
                Must be between min_n and n (inclusive).

        Returns:
            A list of (ngram_tuple, count) pairs sorted by count.

        Raises:
            ValueError: If order is out of range.
        """
        ...

    def items(self, *, order: int | None = None) -> list[tuple[tuple[str, ...], int]]:
        """Return all (n-gram, count) pairs.

        Args:
            order: If specified, only return n-grams of this specific order.
                Must be between min_n and n (inclusive).

        Returns:
            A list of (ngram_tuple, count) pairs.

        Raises:
            ValueError: If order is out of range.
        """
        ...

    def total(self, *, order: int | None = None) -> int:
        """Return the total number of n-gram tokens counted.

        Args:
            order: If specified, return total for this specific order only.
                Must be between min_n and n (inclusive).
                If None, returns the sum across all orders.

        Returns:
            Total count.

        Raises:
            ValueError: If order is out of range.
        """
        ...

    @property
    def n(self) -> int:
        """The n-gram order."""
        ...

    @property
    def min_n(self) -> int:
        """The minimum n-gram order."""
        ...

    def to_counter(self, *, order: int | None = None) -> Counter[tuple[str, ...]]:
        """Convert to a ``collections.Counter``.

        Args:
            order: If specified, only include n-grams of this specific order.
                Must be between min_n and n (inclusive).
                If None, defaults to the highest order (n).

        Returns:
            A Counter mapping n-gram tuples to their counts.

        Raises:
            ValueError: If order is out of range.
        """
        ...

    def clear(self) -> None:
        """Clear all counts."""
        ...

    def __getitem__(self, ngram: Sequence[str]) -> int: ...
    def __len__(self) -> int: ...
    def __contains__(self, ngram: Sequence[str]) -> bool: ...
    def __iter__(self) -> Iterator[tuple[str, ...]]: ...
    def __repr__(self) -> str: ...
    def __add__(self, other: Ngrams) -> Ngrams: ...
    def __iadd__(self, other: Ngrams) -> Ngrams: ...

__all__ = ["Ngrams"]
