"""Tests for rustling.taggers.AveragedPerceptronTagger."""

import os
import tempfile

import pytest

from rustling.taggers import AveragedPerceptronTagger


def test_init_default():
    """Test initialization with default parameters."""
    tagger = AveragedPerceptronTagger()
    assert tagger.classes == set()
    assert tagger.tagdict == {}


def test_init_custom_params():
    """Test initialization with custom parameters."""
    tagger = AveragedPerceptronTagger(
        frequency_threshold=10, ambiguity_threshold=0.95, n_iter=5
    )
    assert tagger.classes == set()
    assert tagger.tagdict == {}


def test_tag_empty():
    """Test tagging an empty list of words."""
    tagger = AveragedPerceptronTagger()
    tags = tagger.tag([])
    assert tags == []


def test_train_and_tag():
    """Test training and tagging."""
    tagger = AveragedPerceptronTagger(
        frequency_threshold=1, ambiguity_threshold=0.9, n_iter=2
    )
    training_data = [
        [("I", "PRON"), ("love", "VERB"), ("cats", "NOUN")],
        [("You", "PRON"), ("love", "VERB"), ("dogs", "NOUN")],
        [("We", "PRON"), ("eat", "VERB"), ("food", "NOUN")],
    ]
    tagger.train(training_data)

    # Check that classes are learned
    assert tagger.classes == {"PRON", "VERB", "NOUN"}

    # Test tagging
    words = ["I", "love", "cats"]
    tags = tagger.tag(words)
    assert len(tags) == 3
    # With enough training, the model should get these right
    assert tags == ["PRON", "VERB", "NOUN"]


def test_save_and_load():
    """Test saving and loading a model."""
    tagger = AveragedPerceptronTagger(
        frequency_threshold=1, ambiguity_threshold=0.9, n_iter=2
    )
    training_data = [
        [("I", "PRON"), ("love", "VERB"), ("cats", "NOUN")],
        [("You", "PRON"), ("love", "VERB"), ("dogs", "NOUN")],
    ]
    tagger.train(training_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.json")
        tagger.save(model_path)

        # Load into a new tagger
        new_tagger = AveragedPerceptronTagger()
        new_tagger.load(model_path)

        # Verify loaded tagger has the same state
        assert new_tagger.classes == tagger.classes

        # Verify loaded tagger works
        words = ["I", "love", "dogs"]
        original_tags = tagger.tag(words)
        loaded_tags = new_tagger.tag(words)
        assert loaded_tags == original_tags


def test_load_nonexistent_file():
    """Test that loading a nonexistent file raises FileNotFoundError."""
    tagger = AveragedPerceptronTagger()
    with pytest.raises(FileNotFoundError, match="Can't locate tagger model"):
        tagger.load("/nonexistent/path/model.json")


def test_weights_property():
    """Test the weights property."""
    tagger = AveragedPerceptronTagger(
        frequency_threshold=1, ambiguity_threshold=0.9, n_iter=2
    )
    training_data = [
        [("hello", "NOUN"), ("world", "NOUN")],
    ]
    tagger.train(training_data)

    weights = tagger.weights
    assert isinstance(weights, dict)


def test_tagdict_property():
    """Test the tagdict property."""
    tagger = AveragedPerceptronTagger(
        frequency_threshold=1, ambiguity_threshold=0.9, n_iter=2
    )
    training_data = [
        [("hello", "NOUN"), ("world", "NOUN")],
        [("hello", "NOUN"), ("there", "ADV")],
    ]
    tagger.train(training_data)

    tagdict = tagger.tagdict
    assert isinstance(tagdict, dict)
    # "hello" appears twice with NOUN, should be in tagdict
    assert "hello" in tagdict
    assert tagdict["hello"] == "NOUN"
