"""Type stubs for rustling.taggers."""

from __future__ import annotations

from typing import Sequence

class AveragedPerceptronTagger:
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
    ) -> None:
        """Initialize a part-of-speech tagger.

        Parameters
        ----------
        frequency_threshold : int, optional
            A good number of words are almost unambiguously associated with
            a given tag. If these words have a frequency of occurrence above
            this threshold in the training data, they are directly associated
            with their tag in the model.
        ambiguity_threshold : float, optional
            A good number of words are almost unambiguously associated with
            a given tag. If the ratio of (# of occurrences of this word with
            this tag) / (# of occurrences of this word) in the training data
            is equal to or greater than this threshold, then this word is
            directly associated with the tag in the model.
        n_iter : int, optional
            Number of times the training phase iterates through the data.
            At each new iteration, the data is randomly shuffled.
        """
        ...

    def tag(self, words: Sequence[str]) -> list[str]:
        """Tag the words.

        Parameters
        ----------
        words : Sequence[str]
            A segmented sentence or phrase, where each word is a string.

        Returns
        -------
        list[str]
            The list of predicted tags.
        """
        ...

    def train(self, tagged_sents: Sequence[Sequence[tuple[str, str]]]) -> None:
        """Train a model.

        Parameters
        ----------
        tagged_sents : Sequence[Sequence[tuple[str, str]]]
            A list of segmented and tagged sentences for training.
            Each sentence is a sequence of (word, tag) tuples.
        """
        ...

    def save(self, path: str) -> None:
        """Save the model to a JSON file.

        Parameters
        ----------
        path : str
            The path where the model will be saved as a JSON file.
        """
        ...

    def load(self, path: str) -> None:
        """Load a model from a JSON file.

        Parameters
        ----------
        path : str
            The path where the model, stored as a JSON file, is located.

        Raises
        ------
        FileNotFoundError
            If the file at the given path does not exist.
        EnvironmentError
            If the file cannot be read as a tagger model.
        """
        ...

    @property
    def weights(self) -> dict[str, dict[str, float]]:
        """Get the model's weights dictionary.

        Returns
        -------
        dict[str, dict[str, float]]
            A dictionary mapping features to their weight vectors.
        """
        ...

    @property
    def tagdict(self) -> dict[str, str]:
        """Get the tag dictionary.

        Returns
        -------
        dict[str, str]
            A dictionary mapping words to their most likely tags.
        """
        ...

    @property
    def classes(self) -> set[str]:
        """Get the set of POS tag classes.

        Returns
        -------
        set[str]
            A set of all tag classes in the model.
        """
        ...
