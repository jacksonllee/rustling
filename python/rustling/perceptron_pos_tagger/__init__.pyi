"""Averaged perceptron part-of-speech tagging."""

from __future__ import annotations

import os
from typing import Sequence

from rustling.seq_feature import SeqFeatureTemplate

class AveragedPerceptron:
    """A part-of-speech tagger using an averaged perceptron model.

    This is a modified version based on the textblob-aptagger codebase
    (MIT license), with original implementation by Matthew Honnibal.
    """

    def __init__(
        self,
        *,
        frequency_threshold: int = 10,
        ambiguity_threshold: float = 0.95,
        n_iter: int = 5,
        random_seed: int | None = None,
        features: Sequence[SeqFeatureTemplate] | None = None,
    ) -> None:
        """Initialize a part-of-speech tagger.

        Args:
            frequency_threshold: A good number of words are almost
                unambiguously associated with a given tag. If these words
                have a frequency of occurrence above this threshold in the
                training data, they are directly associated with their tag
                in the model.
            ambiguity_threshold: A good number of words are almost
                unambiguously associated with a given tag. If the ratio of
                (# of occurrences of this word with this tag) /
                (# of occurrences of this word) in the training data is
                equal to or greater than this threshold, then this word is
                directly associated with the tag in the model.
            n_iter: Number of times the training phase iterates through
                the data. At each new iteration, the data is randomly
                shuffled.
            random_seed: Random seed for reproducible shuffling during
                training. If ``None``, a non-deterministic random number
                generator is used.
            features: Optional list of feature templates created with
                ``seq_obs()`` and ``seq_label()``. If ``None``, uses
                default features.
        """
        ...

    def predict(self, sequences: Sequence[Sequence[str]]) -> list[list[str]]:
        """Predict tags for the sequences.

        Args:
            sequences: A list of segmented sentences, where each sentence
                is a sequence of words.

        Returns:
            A list of tag sequences, one per input sentence.
        """
        ...

    def fit(
        self,
        sequences: Sequence[Sequence[str]],
        tags: Sequence[Sequence[str]],
    ) -> None:
        """Fit a model.

        Args:
            sequences: A list of segmented sentences for training, where
                each sentence is a sequence of words.
            tags: A list of tag sequences corresponding to the sentences.
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

        Raises:
            FileNotFoundError: If the file at the given path does not
                exist.
            EnvironmentError: If the file cannot be read as a tagger
                model.
        """
        ...

    @property
    def weights(self) -> dict[str, dict[str, float]]:
        """Get the model's weights dictionary.

        Returns:
            A dictionary mapping features to their weight vectors.
        """
        ...

    @property
    def tagdict(self) -> dict[str, str]:
        """Get the tag dictionary.

        Returns:
            A dictionary mapping words to their most likely tags.
        """
        ...

    @property
    def classes(self) -> set[str]:
        """Get the set of POS tag classes.

        Returns:
            A set of all tag classes in the model.
        """
        ...
