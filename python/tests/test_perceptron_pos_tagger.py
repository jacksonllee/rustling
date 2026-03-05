"""Tests for rustling.perceptron_pos_tagger.AveragedPerceptron."""

import os
import tempfile

import pytest

from rustling.perceptron_pos_tagger import AveragedPerceptron


def test_init_default():
    """Test initialization with default parameters."""
    tagger = AveragedPerceptron()
    assert tagger.classes == set()
    assert tagger.tagdict == {}


def test_init_custom_params():
    """Test initialization with custom parameters."""
    tagger = AveragedPerceptron(
        frequency_threshold=10, ambiguity_threshold=0.95, n_iter=5
    )
    assert tagger.classes == set()
    assert tagger.tagdict == {}


def test_predict_empty():
    """Test predicting on an empty list of sequences."""
    tagger = AveragedPerceptron()
    result = tagger.predict([])
    assert result == []

    result = tagger.predict([[]])
    assert result == [[]]


def test_fit_and_predict():
    """Test fitting and predicting."""
    tagger = AveragedPerceptron(
        frequency_threshold=1, ambiguity_threshold=0.9, n_iter=2
    )
    sequences = [
        ["I", "love", "cats"],
        ["You", "love", "dogs"],
        ["We", "eat", "food"],
    ]
    tags = [
        ["PRON", "VERB", "NOUN"],
        ["PRON", "VERB", "NOUN"],
        ["PRON", "VERB", "NOUN"],
    ]
    tagger.fit(sequences, tags)

    # Check that classes are learned
    assert tagger.classes == {"PRON", "VERB", "NOUN"}

    # Test tagging
    result = tagger.predict([["I", "love", "cats"]])
    assert len(result) == 1
    assert len(result[0]) == 3
    # With enough training, the model should get these right
    assert result[0] == ["PRON", "VERB", "NOUN"]


def test_save_and_load():
    """Test saving and loading a model."""
    tagger = AveragedPerceptron(
        frequency_threshold=1, ambiguity_threshold=0.9, n_iter=2
    )
    sequences = [
        ["I", "love", "cats"],
        ["You", "love", "dogs"],
    ]
    tags = [
        ["PRON", "VERB", "NOUN"],
        ["PRON", "VERB", "NOUN"],
    ]
    tagger.fit(sequences, tags)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.json")
        tagger.save(model_path)

        # Load into a new tagger
        new_tagger = AveragedPerceptron()
        new_tagger.load(model_path)

        # Verify loaded tagger has the same state
        assert new_tagger.classes == tagger.classes

        # Verify loaded tagger works
        words = [["I", "love", "dogs"]]
        original_tags = tagger.predict(words)
        loaded_tags = new_tagger.predict(words)
        assert loaded_tags == original_tags


def test_load_nonexistent_file():
    """Test that loading a nonexistent file raises FileNotFoundError."""
    tagger = AveragedPerceptron()
    with pytest.raises(FileNotFoundError, match="Can't locate tagger model"):
        tagger.load("/nonexistent/path/model.json")


def test_weights_property():
    """Test the weights property."""
    tagger = AveragedPerceptron(
        frequency_threshold=1, ambiguity_threshold=0.9, n_iter=2
    )
    sequences = [
        ["hello", "world"],
    ]
    tags = [
        ["NOUN", "NOUN"],
    ]
    tagger.fit(sequences, tags)

    weights = tagger.weights
    assert isinstance(weights, dict)


def test_tagdict_property():
    """Test the tagdict property."""
    tagger = AveragedPerceptron(
        frequency_threshold=1, ambiguity_threshold=0.9, n_iter=2
    )
    sequences = [
        ["hello", "world"],
        ["hello", "there"],
    ]
    tags = [
        ["NOUN", "NOUN"],
        ["NOUN", "ADV"],
    ]
    tagger.fit(sequences, tags)

    tagdict = tagger.tagdict
    assert isinstance(tagdict, dict)
    # "hello" appears twice with NOUN, should be in tagdict
    assert "hello" in tagdict
    assert tagdict["hello"] == "NOUN"
