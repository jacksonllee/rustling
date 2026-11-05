import pytest

from rustling.hmm import HiddenMarkovModel

EMPTY_BATCHES = [
    ([], None),
    ([[]], None),
    ([[], []], None),
    ([], []),
    ([[]], [[]]),
    ([[], []], [[], []]),
]
TRAINING = [["a", "b", "a"], ["b", "a", "b"]]


@pytest.mark.parametrize("sequences, labels", EMPTY_BATCHES)
def test_empty_fit_allows_later_cold_training(sequences, labels):
    model = HiddenMarkovModel(n_states=2, n_iter=3, random_seed=42)
    model.fit(sequences, labels=labels)

    with pytest.raises(ValueError, match="not been fitted"):
        model.score(TRAINING)
    with pytest.raises(ValueError, match="not been fitted"):
        model.predict(TRAINING)

    model.fit(TRAINING)
    reference = HiddenMarkovModel(n_states=2, n_iter=3, random_seed=42)
    reference.fit(TRAINING)
    assert model.score(TRAINING) == reference.score(TRAINING)
    assert model.predict(TRAINING) == reference.predict(TRAINING)


@pytest.mark.parametrize("sequences, labels", EMPTY_BATCHES)
def test_empty_fit_preserves_trained_model(sequences, labels):
    model = HiddenMarkovModel(n_states=2, n_iter=3, random_seed=42)
    model.fit(TRAINING, labels=[["X", "Y", "X"], ["Y", "X", "Y"]])
    scores = model.score(TRAINING)
    predictions = model.predict(TRAINING)

    model.fit(sequences, labels=labels)

    assert model.score(TRAINING) == scores
    assert model.predict(TRAINING) == predictions
