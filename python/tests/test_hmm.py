"""Tests for rustling.hmm.HiddenMarkovModel."""

import math
import os
import tempfile

import pytest

from rustling.hmm import HiddenMarkovModel


def toy_sequences():
    return [
        ["a", "b", "a", "a", "b"],
        ["b", "a", "b", "b"],
        ["a", "a", "a", "b"],
    ]


def test_init_default():
    """Test initialization with default parameters."""
    model = HiddenMarkovModel(n_states=2)
    assert model.n_states == 2


def test_init_custom():
    """Test initialization with custom parameters."""
    model = HiddenMarkovModel(n_states=3, n_iter=50, tolerance=1e-4, random_seed=42)
    assert model.n_states == 3


def test_invalid_n_states():
    """Test that n_states < 1 raises ValueError."""
    with pytest.raises(ValueError, match="n_states must be >= 1"):
        HiddenMarkovModel(n_states=0)


def test_invalid_n_iter():
    """Test that n_iter < 1 raises ValueError."""
    with pytest.raises(ValueError, match="n_iter must be >= 1"):
        HiddenMarkovModel(n_states=2, n_iter=0)


def test_invalid_tolerance():
    """Test that tolerance < 0 raises ValueError."""
    with pytest.raises(ValueError, match="tolerance must be >= 0"):
        HiddenMarkovModel(n_states=2, tolerance=-1.0)


def test_invalid_gamma_zero():
    """Test that gamma=0 raises ValueError."""
    with pytest.raises(ValueError, match="gamma must be > 0"):
        HiddenMarkovModel(n_states=2, gamma=0.0)


def test_invalid_gamma_negative():
    """Test that gamma < 0 raises ValueError."""
    with pytest.raises(ValueError, match="gamma must be > 0"):
        HiddenMarkovModel(n_states=2, gamma=-1.0)


def test_gamma_custom():
    """Test supervised training with a custom gamma."""
    model = HiddenMarkovModel(n_states=2, gamma=0.5, random_seed=42)
    sequences = [["a", "b", "a"], ["b", "a", "b"]]
    labels = [["X", "Y", "X"], ["Y", "X", "Y"]]
    model.fit(sequences, labels=labels)
    result = model.predict([["a", "b"]])
    assert len(result) == 1
    assert len(result[0]) == 2


def test_predict_before_fit():
    """Test that predicting before fitting raises ValueError."""
    model = HiddenMarkovModel(n_states=2)
    with pytest.raises(ValueError, match="not been fitted"):
        model.predict([["a", "b"]])


def test_score_before_fit():
    """Test that scoring before fitting raises ValueError."""
    model = HiddenMarkovModel(n_states=2)
    with pytest.raises(ValueError, match="not been fitted"):
        model.score([["a", "b"]])


def test_fit_and_predict():
    """Test fitting and predicting."""
    model = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    model.fit(toy_sequences())
    result = model.predict([["a", "b", "a"]])
    assert len(result) == 1
    assert len(result[0]) == 3
    assert all(0 <= s < 2 for s in result[0])


def test_predict_empty():
    """Test predicting on an empty sequence."""
    model = HiddenMarkovModel(n_states=2, random_seed=42)
    model.fit(toy_sequences())
    result = model.predict([[]])
    assert result == [[]]


def test_score_returns_finite():
    """Test that score returns a finite negative value."""
    model = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    model.fit(toy_sequences())
    scores = model.score([["a", "b", "a"]])
    assert len(scores) == 1
    assert math.isfinite(scores[0])
    assert scores[0] < 0  # log-likelihood is negative for probabilities < 1


def test_score_empty():
    """Test scoring an empty sequence returns 0.0."""
    model = HiddenMarkovModel(n_states=2, random_seed=42)
    model.fit(toy_sequences())
    scores = model.score([[]])
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_deterministic_with_seed():
    """Test that the same seed produces the same results."""
    model1 = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    model1.fit(toy_sequences())
    model2 = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    model2.fit(toy_sequences())
    assert model1.predict([["a", "b"]]) == model2.predict([["a", "b"]])
    assert model1.score([["a", "b"]]) == model2.score([["a", "b"]])


def test_predict_unknown_observation():
    """Test predicting with an unknown observation."""
    model = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    model.fit(toy_sequences())
    # "c" was not in training data
    result = model.predict([["a", "c", "b"]])
    assert len(result) == 1
    assert len(result[0]) == 3


def test_score_unknown_observation():
    """Test scoring with an unknown observation."""
    model = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    model.fit(toy_sequences())
    scores = model.score([["a", "c", "b"]])
    assert len(scores) == 1
    assert math.isfinite(scores[0])


def test_fit_with_empty_sequences():
    """Empty sequences in training data should be silently ignored."""
    model = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    data = toy_sequences() + [[]]  # add an empty sequence
    model.fit(data)
    result = model.predict([["a", "b"]])
    assert len(result) == 1
    assert len(result[0]) == 2


def test_save_and_load():
    """Test saving and loading a model."""
    model = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    model.fit(toy_sequences())

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "hmm_model.json")
        model.save(model_path)

        # Load into a new model
        new_model = HiddenMarkovModel(n_states=2)
        new_model.load(model_path)

        # Verify loaded model works and gives same predictions
        obs = [["a", "b", "a"]]
        assert new_model.predict(obs) == model.predict(obs)
        # Scores may differ slightly due to emission value rounding during save.
        assert new_model.score(obs)[0] == pytest.approx(model.score(obs)[0], abs=0.1)


def test_load_nonexistent_file():
    """Test that loading a nonexistent file raises FileNotFoundError."""
    model = HiddenMarkovModel(n_states=2)
    with pytest.raises(FileNotFoundError, match="Can't locate HMM model"):
        model.load("/nonexistent/path/model.json")


def test_semi_supervised_basic():
    """Test supervised fit followed by unsupervised fit."""
    model = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    sequences = [
        ["a", "b", "a", "a", "b"],
        ["b", "a", "b", "b"],
    ]
    labels = [
        ["X", "Y", "X", "X", "Y"],
        ["Y", "X", "Y", "Y"],
    ]
    model.fit(sequences, labels=labels)

    # Unsupervised refinement.
    unlabeled = [["a", "b", "a"], ["b", "a", "b"]]
    model.fit(unlabeled)

    # Predict should still work and return valid state indices.
    result = model.predict([["a", "b", "a"]])
    assert len(result) == 1
    assert len(result[0]) == 3
    assert all(0 <= s < 2 for s in result[0])


def test_semi_supervised_new_vocab():
    """Test semi-supervised training with new observations in unsupervised step."""
    model = HiddenMarkovModel(n_states=2, n_iter=10, random_seed=42)
    sequences = [["a", "b", "a"]]
    labels = [["X", "Y", "X"]]
    model.fit(sequences, labels=labels)

    # Unsupervised fit introduces "c" not seen during supervised training.
    unlabeled = [["a", "c", "b"], ["c", "a"]]
    model.fit(unlabeled)

    # Predict handles the new observation.
    result = model.predict([["c", "a", "b"]])
    assert len(result) == 1
    assert len(result[0]) == 3
    assert all(0 <= s < 2 for s in result[0])
