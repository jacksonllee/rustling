"""Tests for rustling.wordseg.RandomSegmenter."""

import pytest

from rustling.wordseg import RandomSegmenter


def test_valid_prob_zero():
    """Test that prob=0.0 is valid and produces no segmentation."""
    segmenter = RandomSegmenter(prob=0.0)
    result = segmenter.predict(["hello"])
    # With prob=0.0, no segmentation should occur
    assert result == [["hello"]]


def test_valid_prob_half():
    """Test that prob=0.5 is valid."""
    segmenter = RandomSegmenter(prob=0.5)
    # Just verify it runs without error
    result = segmenter.predict(["hello"])
    assert len(result) == 1
    # The segments should join back to the original
    assert "".join(result[0]) == "hello"


def test_invalid_prob_negative():
    """Test that negative prob raises ValueError."""
    with pytest.raises(ValueError, match="prob must be from"):
        RandomSegmenter(prob=-0.1)


def test_invalid_prob_one():
    """Test that prob=1.0 raises ValueError."""
    with pytest.raises(ValueError, match="prob must be from"):
        RandomSegmenter(prob=1.0)


def test_invalid_prob_greater_than_one():
    """Test that prob > 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="prob must be from"):
        RandomSegmenter(prob=1.5)


def test_fit_raises_error():
    """Test that fit raises NotImplementedError."""
    segmenter = RandomSegmenter(prob=0.5)
    with pytest.raises(NotImplementedError, match="No training needed"):
        segmenter.fit([])


def test_empty_input():
    """Test segmentation of empty string."""
    segmenter = RandomSegmenter(prob=0.5)
    result = segmenter.predict([""])
    assert result == [[]]


def test_single_char():
    """Test segmentation of single character."""
    segmenter = RandomSegmenter(prob=0.5)
    result = segmenter.predict(["a"])
    # Single char cannot be segmented further
    assert result == [["a"]]


def test_unicode():
    """Test segmentation of Unicode characters."""
    segmenter = RandomSegmenter(prob=0.0)
    result = segmenter.predict(["你好"])
    # With prob=0.0, no segmentation should occur
    assert result == [["你好"]]


def test_multiple_sentences():
    """Test segmentation of multiple sentences."""
    segmenter = RandomSegmenter(prob=0.0)
    result = segmenter.predict(["hello", "world"])
    assert result == [["hello"], ["world"]]


def test_segments_preserve_content():
    """Test that segmentation preserves original content."""
    segmenter = RandomSegmenter(prob=0.5)
    inputs = ["hello", "world", "test123"]
    results = segmenter.predict(inputs)

    for inp, segments in zip(inputs, results):
        # Join segments back together
        assert "".join(segments) == inp
