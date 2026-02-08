"""Tests for rustling.wordseg.LongestStringMatching."""

import pytest

from rustling.wordseg import LongestStringMatching


def test_basic():
    """Test basic segmentation from the original wordseg package."""
    model = LongestStringMatching(max_word_length=4)
    model.fit(
        [
            ("this", "is", "a", "sentence"),
            ("that", "is", "not", "a", "sentence"),
        ]
    )
    result = model.predict(["thatisadog", "thisisnotacat"])
    assert result == [
        ["that", "is", "a", "d", "o", "g"],
        ["this", "is", "not", "a", "c", "a", "t"],
    ]


def test_invalid_max_word_length():
    """Test that max_word_length < 2 raises ValueError."""
    with pytest.raises(ValueError, match="max_word_length must be >= 2"):
        LongestStringMatching(max_word_length=1)


def test_empty_input():
    """Test segmentation of empty string."""
    model = LongestStringMatching(max_word_length=4)
    model.fit([["hello", "world"]])
    result = model.predict([""])
    assert result == [[]]


def test_no_training_data():
    """Test segmentation with no training data falls back to single chars."""
    model = LongestStringMatching(max_word_length=4)
    model.fit([])
    result = model.predict(["hello"])
    assert result == [["h", "e", "l", "l", "o"]]


def test_single_char_words_ignored_in_training():
    """Test that single-character words are ignored in training."""
    model = LongestStringMatching(max_word_length=4)
    # Single-character words should be ignored in training
    model.fit([["a", "b", "ab"]])
    # "ab" should be recognized, but not single chars
    result = model.predict(["abab"])
    assert result == [["ab", "ab"]]


def test_unicode_chars():
    """Test segmentation of Unicode characters (e.g., Chinese)."""
    model = LongestStringMatching(max_word_length=4)
    model.fit([["你好", "世界"]])
    result = model.predict(["你好世界"])
    assert result == [["你好", "世界"]]


def test_max_word_length_constraint():
    """Test that max_word_length limits matching length."""
    model = LongestStringMatching(max_word_length=3)
    # Train with a word longer than max_word_length
    model.fit([["hello"]])
    # Even though "hello" is in training, we can only match up to 3 chars
    # "hel" is not in training, so we fall back character by character
    result = model.predict(["hello"])
    assert result == [["h", "e", "l", "l", "o"]]


def test_accepts_tuples_in_fit():
    """Test that fit accepts tuples (like the original Python wordseg)."""
    model = LongestStringMatching(max_word_length=4)
    model.fit(
        [
            ("this", "is"),
            ("that", "was"),
        ]
    )
    result = model.predict(["thisis"])
    assert result == [["this", "is"]]


def test_multiple_sentences():
    """Test segmentation of multiple sentences at once."""
    model = LongestStringMatching(max_word_length=4)
    model.fit([["the", "cat"]])
    result = model.predict(["thecat", "catthe", "the"])
    assert result == [["the", "cat"], ["cat", "the"], ["the"]]
