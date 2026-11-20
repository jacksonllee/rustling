"""Tests for rustling.wordseg.HiddenMarkovModelSegmenter."""

import pytest

from rustling.wordseg import HiddenMarkovModelSegmenter


def test_init():
    """Test initialization."""
    HiddenMarkovModelSegmenter()


def test_predict_before_fit():
    """Test that predict before fitting falls back to single characters."""
    segmenter = HiddenMarkovModelSegmenter()
    result = segmenter.predict(["hello"])
    assert result == [list("hello")]


def test_predict_empty():
    """Test predicting on empty input."""
    segmenter = HiddenMarkovModelSegmenter()
    segmenter.fit_segmented([["hello", "world"]])
    result = segmenter.predict([])
    assert result == []


def test_predict_empty_string():
    """Test predicting on an empty string."""
    segmenter = HiddenMarkovModelSegmenter()
    segmenter.fit_segmented([["hello", "world"]])
    result = segmenter.predict([""])
    assert result == [[]]


def test_fit_and_predict():
    """Test fitting and predicting."""
    segmenter = HiddenMarkovModelSegmenter()
    training_data = [
        ["this", "is", "a", "test"],
        ["that", "is", "not", "a", "test"],
    ] * 10
    segmenter.fit_segmented(training_data)

    result = segmenter.predict(["thisisa"])
    assert len(result) == 1
    assert "".join(result[0]) == "thisisa"


def test_fit_and_predict_chinese():
    """Test fitting and predicting with Chinese characters."""
    segmenter = HiddenMarkovModelSegmenter()
    training_data = [["你好", "世界"]] * 20
    segmenter.fit_segmented(training_data)

    result = segmenter.predict(["你好世界"])
    assert len(result) == 1
    assert "".join(result[0]) == "你好世界"


def test_deterministic():
    """Test that results are deterministic."""
    training_data = [
        ["ab", "cd"],
        ["ef", "gh"],
    ]

    seg1 = HiddenMarkovModelSegmenter()
    seg1.fit_segmented(training_data)
    result1 = seg1.predict(["abcd"])

    seg2 = HiddenMarkovModelSegmenter()
    seg2.fit_segmented(training_data)
    result2 = seg2.predict(["abcd"])

    assert result1 == result2


def test_accepts_tuples_in_fit():
    """Test that fit accepts tuples."""
    segmenter = HiddenMarkovModelSegmenter()
    segmenter.fit_segmented(
        [
            ("this", "is"),
            ("that", "was"),
        ]
    )
    result = segmenter.predict(["thisis"])
    assert len(result) == 1
    assert "".join(result[0]) == "thisis"


def test_multiple_sentences():
    """Test segmentation of multiple sentences at once."""
    segmenter = HiddenMarkovModelSegmenter()
    training_data = [["ab", "cd"]] * 20
    segmenter.fit_segmented(training_data)
    result = segmenter.predict(["abcd", "cdab"])
    assert len(result) == 2
    assert "".join(result[0]) == "abcd"
    assert "".join(result[1]) == "cdab"


def test_init_with_gamma():
    """Test initialization with custom gamma."""
    segmenter = HiddenMarkovModelSegmenter(gamma=0.5)
    training_data = [["hello", "world"]] * 10
    segmenter.fit_segmented(training_data)
    result = segmenter.predict(["helloworld"])
    assert len(result) == 1
    assert "".join(result[0]) == "helloworld"


@pytest.mark.parametrize("empty", [[], [[]], [[""]]])
def test_fit_empty_data_keeps_untrained_fallback(empty):
    """Fitting on no usable data should leave a fresh segmenter untrained."""
    model = HiddenMarkovModelSegmenter(n_iter=3, random_seed=42)
    model.fit_segmented(empty)

    assert model.predict(["你好世界"]) == [list("你好世界")]
    assert model.predict(["你好"], offsets=True) == [[("你", (0, 1)), ("好", (1, 2))]]
    with pytest.raises(ValueError, match="not been fitted"):
        model.score([["你好", "世界"]])

    training = ["你好世界", "世界你好"]
    model.fit_unsegmented(training)
    reference = HiddenMarkovModelSegmenter(n_iter=3, random_seed=42)
    reference.fit_unsegmented(training)
    assert model.score([["你好", "世界"]]) == reference.score([["你好", "世界"]])
    assert model.predict(training) == reference.predict(training)


@pytest.mark.parametrize("empty", [[], [[]], [[""]]])
def test_fit_empty_data_preserves_trained_model(empty):
    """Fitting on no usable data should leave a trained segmenter unchanged."""
    model = HiddenMarkovModelSegmenter(n_iter=3, random_seed=42)
    training = [["你好", "世界"], ["世界", "你好"]]
    model.fit_segmented(training)
    before = model.score(training), model.predict(["你好世界"], offsets=True)

    model.fit_segmented(empty)

    assert (model.score(training), model.predict(["你好世界"], offsets=True)) == before


def test_predict_offsets():
    """Test predict with offsets=True returns (word, (start, end)) tuples."""
    segmenter = HiddenMarkovModelSegmenter()
    training_data = [["你好", "世界"]] * 20
    segmenter.fit_segmented(training_data)
    input_str = "你好世界"
    result = segmenter.predict([input_str], offsets=True)
    assert len(result) == 1
    for word, (start, end) in result[0]:
        assert input_str[start:end] == word


def test_predict_offsets_false_unchanged():
    """Test that offsets=False returns plain strings."""
    segmenter = HiddenMarkovModelSegmenter()
    training_data = [["你好", "世界"]] * 20
    segmenter.fit_segmented(training_data)
    result = segmenter.predict(["你好世界"], offsets=False)
    assert all(isinstance(w, str) for w in result[0])


def test_predict_offsets_default_unchanged():
    """Test that default (no offsets kwarg) returns plain strings."""
    segmenter = HiddenMarkovModelSegmenter()
    training_data = [["你好", "世界"]] * 20
    segmenter.fit_segmented(training_data)
    result = segmenter.predict(["你好世界"])
    assert all(isinstance(w, str) for w in result[0])


def test_predict_offsets_empty():
    """Test predict with offsets=True on empty input."""
    segmenter = HiddenMarkovModelSegmenter()
    assert segmenter.predict([], offsets=True) == []


def test_predict_offsets_empty_string():
    """Test predict with offsets=True on empty string."""
    segmenter = HiddenMarkovModelSegmenter()
    assert segmenter.predict([""], offsets=True) == [[]]
