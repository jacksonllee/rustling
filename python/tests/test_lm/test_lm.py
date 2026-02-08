"""Tests for rustling.lm language models."""

import math

import pytest

from rustling.lm import LanguageModel, MLE, Lidstone, Laplace


def train_data():
    """Simple training data for tests."""
    return [
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
        ["the", "cat", "ran"],
    ]


# --- LanguageModel base tests ---


class TestLanguageModel:
    def test_init_mle(self):
        model = LanguageModel(order=2, smoothing="mle")
        assert model.order == 2

    def test_init_lidstone(self):
        model = LanguageModel(order=2, smoothing="lidstone", gamma=0.5)
        assert model.order == 2

    def test_init_laplace(self):
        model = LanguageModel(order=2, smoothing="laplace")
        assert model.order == 2

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            LanguageModel(order=0, smoothing="mle")

    def test_invalid_smoothing(self):
        with pytest.raises(ValueError, match="Unknown smoothing"):
            LanguageModel(order=2, smoothing="unknown")

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            LanguageModel(order=2, smoothing="lidstone", gamma=-1.0)

    def test_score_before_fit(self):
        model = LanguageModel(order=2, smoothing="mle")
        with pytest.raises(ValueError, match="not been fitted"):
            model.score("cat", ["the"])

    def test_generate_before_fit(self):
        model = LanguageModel(order=2, smoothing="mle")
        with pytest.raises(ValueError, match="not been fitted"):
            model.generate(num_words=5)

    def test_vocab_size(self):
        model = LanguageModel(order=2, smoothing="mle")
        model.fit(train_data())
        # 5 words (the, cat, sat, dog, ran) + 3 special (<UNK>, <s>, </s>)
        assert model.vocab_size == 8


# --- MLE tests ---


class TestMLE:
    def test_init(self):
        model = MLE(order=2)
        assert model.order == 2

    def test_isinstance(self):
        model = MLE(order=2)
        assert isinstance(model, LanguageModel)

    def test_bigram_score(self):
        model = MLE(order=2)
        model.fit(train_data())
        # P(cat | the) = count(the, cat) / count(the, *)
        # count(the, cat) = 2, count(the, *) = 3 (cat x2, dog x1)
        score = model.score("cat", ["the"])
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_bigram_score_dog(self):
        model = MLE(order=2)
        model.fit(train_data())
        # P(dog | the) = 1/3
        score = model.score("dog", ["the"])
        assert abs(score - 1.0 / 3.0) < 1e-9

    def test_unseen_bigram_is_zero(self):
        model = MLE(order=2)
        model.fit(train_data())
        # "fish" is OOV -> mapped to <UNK>, P(<UNK> | the) = 0
        score = model.score("fish", ["the"])
        assert score == 0.0

    def test_unigram(self):
        model = MLE(order=1)
        model.fit(train_data())
        # P(the) = count(the) / total_unigram_count
        score = model.score("the", [])
        assert score > 0.0

    def test_unigram_no_context(self):
        model = MLE(order=1)
        model.fit(train_data())
        # score without context arg
        score = model.score("the")
        assert score > 0.0

    def test_logscore(self):
        model = MLE(order=2)
        model.fit(train_data())
        score = model.score("cat", ["the"])
        logscore = model.logscore("cat", ["the"])
        assert abs(logscore - math.log2(score)) < 1e-9

    def test_logscore_zero_is_neg_inf(self):
        model = MLE(order=2)
        model.fit(train_data())
        logscore = model.logscore("fish", ["the"])
        assert logscore == float("-inf")

    def test_score_vs_unmasked_score(self):
        model = MLE(order=2)
        model.fit(train_data())
        # In-vocab word: same result
        s1 = model.score("cat", ["the"])
        s2 = model.unmasked_score("cat", ["the"])
        assert abs(s1 - s2) < 1e-9

    def test_accepts_tuples_in_fit(self):
        """Test that fit accepts tuples (like other rustling models)."""
        model = MLE(order=2)
        model.fit([("the", "cat", "sat"), ("the", "dog", "ran")])
        score = model.score("cat", ["the"])
        assert score > 0.0


# --- Lidstone tests ---


class TestLidstone:
    def test_init(self):
        model = Lidstone(order=2, gamma=0.5)
        assert model.order == 2

    def test_isinstance(self):
        model = Lidstone(order=2, gamma=0.5)
        assert isinstance(model, LanguageModel)

    def test_invalid_gamma_zero(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            Lidstone(order=2, gamma=0.0)

    def test_invalid_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            Lidstone(order=2, gamma=-0.5)

    def test_unseen_bigram_nonzero(self):
        model = Lidstone(order=2, gamma=1.0)
        model.fit(train_data())
        # With smoothing, unseen n-grams get nonzero probability
        score = model.score("fish", ["the"])
        assert score > 0.0

    def test_score_formula(self):
        model = Lidstone(order=2, gamma=0.5)
        model.fit(train_data())
        # P(cat | the) = (count(the,cat) + gamma) / (count(the,*) + |V| * gamma)
        # = (2 + 0.5) / (3 + 8 * 0.5) = 2.5 / 7.0
        score = model.score("cat", ["the"])
        assert abs(score - 2.5 / 7.0) < 1e-9

    def test_scores_between_zero_and_one(self):
        model = Lidstone(order=2, gamma=0.5)
        model.fit(train_data())
        score = model.score("cat", ["the"])
        assert 0.0 < score < 1.0


# --- Laplace tests ---


class TestLaplace:
    def test_init(self):
        model = Laplace(order=2)
        assert model.order == 2

    def test_isinstance(self):
        model = Laplace(order=2)
        assert isinstance(model, LanguageModel)

    def test_matches_lidstone_gamma_one(self):
        """Laplace should give same results as Lidstone(gamma=1)."""
        laplace = Laplace(order=2)
        lidstone = Lidstone(order=2, gamma=1.0)
        data = train_data()
        laplace.fit(data)
        lidstone.fit(data)

        for word in ["cat", "dog", "sat", "ran", "fish"]:
            for ctx in [["the"], ["cat"]]:
                assert (
                    abs(laplace.score(word, ctx) - lidstone.score(word, ctx)) < 1e-9
                ), f"Mismatch for word={word}, ctx={ctx}"

    def test_unseen_nonzero(self):
        model = Laplace(order=2)
        model.fit(train_data())
        score = model.score("fish", ["the"])
        assert score > 0.0


# --- Generate tests ---


class TestGenerate:
    def test_generate_returns_list(self):
        model = MLE(order=2)
        model.fit(train_data())
        result = model.generate(num_words=3, random_seed=42)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_generate_deterministic_with_seed(self):
        model = MLE(order=2)
        model.fit(train_data())
        result1 = model.generate(num_words=5, random_seed=42)
        result2 = model.generate(num_words=5, random_seed=42)
        assert result1 == result2

    def test_generate_with_text_seed(self):
        model = MLE(order=2)
        model.fit(train_data())
        result = model.generate(num_words=2, text_seed=["the"], random_seed=42)
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_generate_no_special_tokens(self):
        """Generated words should not include <s> or </s>."""
        model = MLE(order=2)
        model.fit(train_data())
        result = model.generate(num_words=10, random_seed=42)
        for word in result:
            assert word != "<s>"
            assert word != "</s>"

    def test_generate_different_seeds_may_differ(self):
        model = Laplace(order=2)
        model.fit(train_data())
        result1 = model.generate(num_words=20, random_seed=42)
        result2 = model.generate(num_words=20, random_seed=99)
        # Very likely to differ with different seeds and Laplace smoothing
        # (allows more diverse output)
        assert result1 != result2
