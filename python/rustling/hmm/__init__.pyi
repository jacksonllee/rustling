"""Hidden Markov Model."""

from __future__ import annotations

import os
from typing import Sequence

from rustling.seq_feature import SeqFeatureTemplate

class HiddenMarkovModel:
    """A Hidden Markov Model.

    Supports both unsupervised training (Baum-Welch EM algorithm),
    supervised training (label counting with Lidstone smoothing), and
    semi-supervised training.
    Uses the Viterbi algorithm for decoding and the Forward algorithm
    for computing log-likelihoods.
    """

    def __init__(
        self,
        *,
        n_states: int,
        n_iter: int = 100,
        tolerance: float = 1e-6,
        gamma: float = 1.0,
        random_seed: int | None = None,
        features: Sequence[SeqFeatureTemplate] | None = None,
    ) -> None:
        """Initialize a Hidden Markov Model.

        Args:
            n_states: Number of hidden states. Must be >= 1.
                For supervised training (when labels are provided to
                ``fit``), this is auto-set from the number of unique
                labels.
            n_iter: Maximum number of Baum-Welch EM iterations.
                Must be >= 1. Only used for unsupervised training.
            tolerance: Convergence tolerance for the change in
                log-likelihood between EM iterations. Training stops
                early if the change is below this threshold.
                Must be >= 0. Only used for unsupervised training.
            gamma: Lidstone smoothing parameter for supervised
                training. Must be > 0. A value of 1.0 (default)
                corresponds to Laplace (add-one) smoothing.
                Only used for supervised training.
            random_seed: Random seed for reproducible parameter
                initialization. If ``None``, a non-deterministic
                random number generator is used. Only used for
                unsupervised training.
            features: Optional list of observation feature templates
                created with ``seq_obs()``. Only observation features
                are supported (not ``seq_label()``). If ``None``, uses
                a single identity observation feature.

        Raises:
            ValueError: If ``n_states`` < 1, ``n_iter`` < 1,
                ``tolerance`` < 0, ``gamma`` <= 0, or label features
                are provided.
        """
        ...

    def fit(
        self,
        sequences: Sequence[Sequence[str]],
        labels: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """Train the model.

        When ``labels`` are provided, uses supervised counting with
        Lidstone smoothing (configurable via ``gamma``). When
        ``labels`` is ``None``, uses the
        Baum-Welch (EM) algorithm for unsupervised training.

        Semi-supervised training is supported by calling ``fit`` twice:
        first with labels (supervised), then without labels
        (unsupervised). The second call uses the supervised model's
        parameters as the EM initialization instead of random
        initialization, and extends the vocabulary with any new
        observations from the unlabeled data.

        Args:
            sequences: A list of observation sequences. Each sequence
                is a list of observation strings.
            labels: Optional list of label sequences, parallel to
                ``sequences``. Each label sequence must have the same
                length as the corresponding observation sequence.

        Raises:
            ValueError: If sequences and labels have mismatched lengths.
        """
        ...

    def predict(self, sequences: Sequence[Sequence[str]]) -> list[list[int]]:
        """Decode the most likely hidden state sequences.

        Uses the Viterbi algorithm to find the state sequence that
        maximizes the joint probability of the observations and states.
        Unknown observations (not seen during training) are assigned
        a uniform emission probability.

        Args:
            sequences: A list of observation sequences.

        Returns:
            A list of state index lists (0-based) corresponding to the
            most likely hidden state at each time step.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        ...

    def score(self, sequences: Sequence[Sequence[str]]) -> list[float]:
        """Compute the log-likelihood of each observation sequence.

        Uses the Forward algorithm to compute the total log-probability
        of each observation sequence under the model. Unknown observations
        (not seen during training) are assigned a uniform emission
        probability.

        Args:
            sequences: A list of observation sequences.

        Returns:
            A list of log-likelihoods (natural log).

        Raises:
            ValueError: If the model has not been fitted yet.
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
            FileNotFoundError: If the file does not exist.
            EnvironmentError: If the file cannot be read as an HMM model.
        """
        ...

    @property
    def n_states(self) -> int:
        """Number of hidden states."""
        ...
