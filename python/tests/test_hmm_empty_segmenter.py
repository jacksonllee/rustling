import pytest

from rustling.wordseg import HiddenMarkovModelSegmenter


@pytest.mark.parametrize("empty", [[], [[]], [[""]]])
def test_empty_segmented_fit_preserves_cold_fallback_and_later_training(empty):
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
def test_empty_segmented_fit_preserves_trained_model(empty):
    model = HiddenMarkovModelSegmenter(n_iter=3, random_seed=42)
    training = [["你好", "世界"], ["世界", "你好"]]
    model.fit_segmented(training)
    before = model.score(training), model.predict(["你好世界"], offsets=True)

    model.fit_segmented(empty)

    assert (model.score(training), model.predict(["你好世界"], offsets=True)) == before
