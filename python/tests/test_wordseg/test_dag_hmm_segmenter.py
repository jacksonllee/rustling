"""Tests for rustling.wordseg.DAGHMMSegmenter."""

from rustling.wordseg import DAGHMMSegmenter


def _training_data():
    return [
        ["你好", "世界"],
        ["我", "喜歡", "你"],
        ["你好", "我", "喜歡", "世界"],
    ]


def test_fit_and_predict():
    """Test basic fit and predict."""
    seg = DAGHMMSegmenter()
    seg.fit_segmented(_training_data())
    result = seg.predict(["你好世界"])
    assert result == [["你好", "世界"]]


def test_predict_empty():
    """Test predicting on empty input."""
    seg = DAGHMMSegmenter()
    seg.fit_segmented(_training_data())
    assert seg.predict([]) == []


def test_predict_empty_string():
    """Test predicting on an empty string."""
    seg = DAGHMMSegmenter()
    seg.fit_segmented(_training_data())
    assert seg.predict([""]) == [[]]


def test_predict_offsets():
    """Test predict with offsets=True."""
    seg = DAGHMMSegmenter()
    seg.fit_segmented(_training_data())
    input_str = "你好世界"
    result = seg.predict([input_str], offsets=True)
    assert result == [[("你好", (0, 2)), ("世界", (2, 4))]]


def test_predict_offsets_preserves_content():
    """Test that offsets correctly index into the original string."""
    seg = DAGHMMSegmenter()
    seg.fit_segmented(_training_data())
    input_str = "我喜歡你"
    result = seg.predict([input_str], offsets=True)
    assert len(result) == 1
    for word, (start, end) in result[0]:
        assert input_str[start:end] == word


def test_predict_offsets_false_unchanged():
    """Test that offsets=False returns plain strings."""
    seg = DAGHMMSegmenter()
    seg.fit_segmented(_training_data())
    result = seg.predict(["你好世界"], offsets=False)
    assert all(isinstance(w, str) for w in result[0])


def test_predict_offsets_empty():
    """Test predict with offsets=True on empty input."""
    seg = DAGHMMSegmenter()
    seg.fit_segmented(_training_data())
    assert seg.predict([], offsets=True) == []
    assert seg.predict([""], offsets=True) == [[]]
