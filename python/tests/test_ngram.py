"""Tests for rustling.ngram."""

import pytest

from rustling.ngram import Ngrams


class TestNgramsInit:
    def test_init_unigram(self):
        counter = Ngrams(1)
        assert counter.n == 1
        assert len(counter) == 0
        assert counter.total() == 0

    def test_init_bigram(self):
        counter = Ngrams(3)
        assert counter.n == 3

    def test_init_invalid_order(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            Ngrams(0)


class TestNgramsCount:
    def test_count_unigrams(self):
        counter = Ngrams(1)
        counter.count(["the", "cat", "sat", "the"])
        assert counter.get(["the"]) == 2
        assert counter.get(["cat"]) == 1
        assert counter.get(["sat"]) == 1
        assert counter.total() == 4
        assert len(counter) == 3

    def test_count_bigrams(self):
        counter = Ngrams(2)
        counter.count(["the", "cat", "sat", "the", "cat"])
        assert counter.get(["the", "cat"]) == 2
        assert counter.get(["cat", "sat"]) == 1
        assert counter.get(["sat", "the"]) == 1
        assert counter.total() == 4

    def test_count_trigrams(self):
        counter = Ngrams(3)
        counter.count(["a", "b", "c", "a", "b", "c"])
        assert counter.get(["a", "b", "c"]) == 2
        assert counter.get(["b", "c", "a"]) == 1
        assert counter.get(["c", "a", "b"]) == 1
        assert counter.total() == 4

    def test_count_multiple_calls(self):
        counter = Ngrams(1)
        counter.count(["a", "b"])
        counter.count(["b", "c"])
        assert counter.get(["a"]) == 1
        assert counter.get(["b"]) == 2
        assert counter.get(["c"]) == 1
        assert counter.total() == 4

    def test_count_seqs(self):
        counter = Ngrams(1)
        counter.count_seqs([["the", "cat"], ["the", "dog"]])
        assert counter.get(["the"]) == 2
        assert counter.get(["cat"]) == 1
        assert counter.get(["dog"]) == 1
        assert counter.total() == 4

    def test_count_no_cross_boundary(self):
        counter = Ngrams(2)
        counter.count(["a", "b"])
        counter.count(["c", "d"])
        assert counter.get(["b", "c"]) == 0
        assert counter.get(["a", "b"]) == 1
        assert counter.get(["c", "d"]) == 1

    def test_count_short_sentence(self):
        counter = Ngrams(3)
        counter.count(["a", "b"])
        assert counter.total() == 0
        assert len(counter) == 0


class TestNgramsLookup:
    def test_get_existing(self):
        counter = Ngrams(1)
        counter.count(["hello"])
        assert counter.get(["hello"]) == 1

    def test_get_missing(self):
        counter = Ngrams(1)
        assert counter.get(["missing"]) == 0

    def test_getitem(self):
        counter = Ngrams(1)
        counter.count(["hello", "hello", "world"])
        assert counter[["hello"]] == 2
        assert counter[["world"]] == 1
        assert counter[["missing"]] == 0

    def test_contains_true(self):
        counter = Ngrams(1)
        counter.count(["hello"])
        assert ["hello"] in counter

    def test_contains_false(self):
        counter = Ngrams(1)
        assert ["missing"] not in counter


class TestNgramsAggregation:
    def test_most_common_all(self):
        counter = Ngrams(1)
        counter.count(["a", "b", "a", "c", "a", "b"])
        result = counter.most_common()
        assert result[0] == (("a",), 3)
        assert result[1] == (("b",), 2)
        assert result[2] == (("c",), 1)

    def test_most_common_top_n(self):
        counter = Ngrams(1)
        counter.count(["a", "b", "a", "c", "a", "b"])
        result = counter.most_common(2)
        assert len(result) == 2
        assert result[0] == (("a",), 3)
        assert result[1] == (("b",), 2)

    def test_most_common_bigrams(self):
        counter = Ngrams(2)
        counter.count(["the", "cat", "the", "cat", "the", "cat"])
        result = counter.most_common(1)
        # "the cat" x3, "cat the" x2
        assert result[0] == (("the", "cat"), 3)

    def test_items(self):
        counter = Ngrams(1)
        counter.count(["a", "b"])
        items = counter.items()
        items_dict = {ngram: count for ngram, count in items}
        assert items_dict[("a",)] == 1
        assert items_dict[("b",)] == 1
        assert len(items) == 2

    def test_total(self):
        counter = Ngrams(1)
        counter.count(["a", "b", "c"])
        assert counter.total() == 3

    def test_len(self):
        counter = Ngrams(1)
        counter.count(["a", "b", "a"])
        assert len(counter) == 2


class TestNgramsIteration:
    def test_iter(self):
        counter = Ngrams(1)
        counter.count(["a", "b", "c"])
        ngrams = set(counter)
        assert ngrams == {("a",), ("b",), ("c",)}

    def test_iter_bigrams(self):
        counter = Ngrams(2)
        counter.count(["a", "b", "c"])
        ngrams = set(counter)
        assert ngrams == {("a", "b"), ("b", "c")}


class TestNgramsMerge:
    def test_add(self):
        c1 = Ngrams(1)
        c1.count(["a", "b"])
        c2 = Ngrams(1)
        c2.count(["b", "c"])
        merged = c1 + c2
        assert merged.get(["a"]) == 1
        assert merged.get(["b"]) == 2
        assert merged.get(["c"]) == 1
        assert merged.total() == 4

    def test_iadd(self):
        c1 = Ngrams(1)
        c1.count(["a"])
        c2 = Ngrams(1)
        c2.count(["a", "b"])
        c1 += c2
        assert c1.get(["a"]) == 2
        assert c1.get(["b"]) == 1
        assert c1.total() == 3

    def test_add_different_order_raises(self):
        c1 = Ngrams(1)
        c2 = Ngrams(2)
        with pytest.raises(ValueError, match="different orders"):
            c1 + c2

    def test_iadd_different_order_raises(self):
        c1 = Ngrams(1)
        c2 = Ngrams(2)
        with pytest.raises(ValueError, match="different orders"):
            c1 += c2


class TestNgramsClear:
    def test_clear(self):
        counter = Ngrams(1)
        counter.count(["a", "b", "c"])
        assert counter.total() == 3
        counter.clear()
        assert counter.total() == 0
        assert len(counter) == 0
        assert counter.get(["a"]) == 0


class TestNgramsRepr:
    def test_repr(self):
        counter = Ngrams(2)
        counter.count(["a", "b", "c"])
        assert repr(counter) == "Ngrams(n=2, unique=2, total=2)"


class TestNgramsMinN:
    def test_init_with_min_n(self):
        counter = Ngrams(3, min_n=1)
        assert counter.n == 3
        assert counter.min_n == 1
        assert counter.total() == 0

    def test_init_min_n_defaults_to_n(self):
        counter = Ngrams(3)
        assert counter.min_n == 3

    def test_init_min_n_equals_n(self):
        counter = Ngrams(2, min_n=2)
        assert counter.min_n == 2

    def test_init_min_n_zero_raises(self):
        with pytest.raises(ValueError, match="min_n must be >= 1"):
            Ngrams(3, min_n=0)

    def test_init_min_n_greater_than_n_raises(self):
        with pytest.raises(ValueError, match="min_n must be <= n"):
            Ngrams(2, min_n=3)


class TestNgramsAllNgrams:
    def test_count_all_ngrams_basic(self):
        counter = Ngrams(3, min_n=1)
        counter.count(["a", "b", "c"])
        # Unigrams
        assert counter.get(["a"]) == 1
        assert counter.get(["b"]) == 1
        assert counter.get(["c"]) == 1
        # Bigrams
        assert counter.get(["a", "b"]) == 1
        assert counter.get(["b", "c"]) == 1
        # Trigrams
        assert counter.get(["a", "b", "c"]) == 1
        # Total: 3 + 2 + 1 = 6
        assert counter.total() == 6
        assert len(counter) == 6

    def test_count_all_ngrams_per_order_total(self):
        counter = Ngrams(3, min_n=1)
        counter.count(["a", "b", "c"])
        assert counter.total(order=1) == 3
        assert counter.total(order=2) == 2
        assert counter.total(order=3) == 1
        assert counter.total() == 6

    def test_count_all_ngrams_short_sequence(self):
        counter = Ngrams(3, min_n=1)
        counter.count(["a"])
        assert counter.get(["a"]) == 1
        assert counter.total(order=1) == 1
        assert counter.total(order=2) == 0
        assert counter.total(order=3) == 0

    def test_count_all_ngrams_min_n_2(self):
        counter = Ngrams(3, min_n=2)
        counter.count(["a", "b", "c"])
        # No unigrams
        assert counter.get(["a"]) == 0
        # Bigrams
        assert counter.get(["a", "b"]) == 1
        assert counter.get(["b", "c"]) == 1
        # Trigrams
        assert counter.get(["a", "b", "c"]) == 1
        assert counter.total() == 3

    def test_items_with_order_filter(self):
        counter = Ngrams(2, min_n=1)
        counter.count(["a", "b", "c"])
        unigram_items = counter.items(order=1)
        bigram_items = counter.items(order=2)
        assert len(unigram_items) == 3
        assert len(bigram_items) == 2
        all_items = counter.items()
        assert len(all_items) == 5

    def test_most_common_with_order_filter(self):
        counter = Ngrams(2, min_n=1)
        counter.count(["a", "b", "a", "b", "a"])
        # Unigrams: a=3, b=2
        result = counter.most_common(order=1)
        assert result[0] == (("a",), 3)
        assert result[1] == (("b",), 2)
        # Bigrams: top 1
        result = counter.most_common(1, order=2)
        assert len(result) == 1

    def test_total_invalid_order_raises(self):
        counter = Ngrams(3, min_n=2)
        with pytest.raises(ValueError, match="order must be between"):
            counter.total(order=1)
        with pytest.raises(ValueError, match="order must be between"):
            counter.total(order=4)

    def test_items_invalid_order_raises(self):
        counter = Ngrams(3, min_n=2)
        with pytest.raises(ValueError, match="order must be between"):
            counter.items(order=1)

    def test_most_common_invalid_order_raises(self):
        counter = Ngrams(3, min_n=2)
        with pytest.raises(ValueError, match="order must be between"):
            counter.most_common(order=1)

    def test_count_seqs_all_ngrams(self):
        counter = Ngrams(2, min_n=1)
        counter.count_seqs([["a", "b"], ["c", "d"]])
        assert counter.total(order=1) == 4
        assert counter.total(order=2) == 2
        assert counter.get(["b", "c"]) == 0  # no cross-boundary

    def test_iter_all_ngrams(self):
        counter = Ngrams(2, min_n=1)
        counter.count(["a", "b"])
        ngrams = set(counter)
        assert ngrams == {("a",), ("b",), ("a", "b")}

    def test_contains_all_ngrams(self):
        counter = Ngrams(2, min_n=1)
        counter.count(["a", "b"])
        assert ["a"] in counter
        assert ["a", "b"] in counter
        assert ["b", "a"] not in counter


class TestNgramsAllNgramsMerge:
    def test_add_same_config(self):
        c1 = Ngrams(2, min_n=1)
        c1.count(["a", "b"])
        c2 = Ngrams(2, min_n=1)
        c2.count(["b", "c"])
        merged = c1 + c2
        assert merged.get(["b"]) == 2
        assert merged.get(["a", "b"]) == 1
        assert merged.get(["b", "c"]) == 1
        assert merged.total(order=1) == 4
        assert merged.total(order=2) == 2

    def test_add_different_min_n_raises(self):
        c1 = Ngrams(3, min_n=1)
        c2 = Ngrams(3, min_n=2)
        with pytest.raises(ValueError, match="different orders"):
            c1 + c2

    def test_iadd_same_config(self):
        c1 = Ngrams(2, min_n=1)
        c1.count(["a", "b"])
        c2 = Ngrams(2, min_n=1)
        c2.count(["a", "c"])
        c1 += c2
        assert c1.get(["a"]) == 2
        assert c1.total(order=1) == 4
        assert c1.total(order=2) == 2


class TestNgramsAllNgramsClear:
    def test_clear_resets_all_totals(self):
        counter = Ngrams(3, min_n=1)
        counter.count(["a", "b", "c"])
        assert counter.total() == 6
        counter.clear()
        assert counter.total() == 0
        assert counter.total(order=1) == 0
        assert counter.total(order=2) == 0
        assert counter.total(order=3) == 0
        assert len(counter) == 0


class TestNgramsAllNgramsRepr:
    def test_repr_single_order(self):
        counter = Ngrams(2)
        counter.count(["a", "b", "c"])
        assert repr(counter) == "Ngrams(n=2, unique=2, total=2)"

    def test_repr_multi_order(self):
        counter = Ngrams(2, min_n=1)
        counter.count(["a", "b", "c"])
        assert repr(counter) == "Ngrams(n=2, min_n=1, unique=5, total=5)"


class TestNgramsToCounter:
    def test_to_counter_returns_counter(self):
        from collections import Counter

        counter = Ngrams(1)
        counter.count(["a", "b", "a"])
        result = counter.to_counter()
        assert isinstance(result, Counter)

    def test_to_counter_correct_counts(self):
        counter = Ngrams(1)
        counter.count(["a", "b", "a", "c"])
        result = counter.to_counter()
        assert result[("a",)] == 2
        assert result[("b",)] == 1
        assert result[("c",)] == 1
        assert len(result) == 3

    def test_to_counter_bigrams(self):
        counter = Ngrams(2)
        counter.count(["the", "cat", "the", "cat"])
        result = counter.to_counter()
        assert result[("the", "cat")] == 2
        assert result[("cat", "the")] == 1

    def test_to_counter_empty(self):
        from collections import Counter

        counter = Ngrams(1)
        result = counter.to_counter()
        assert result == Counter()

    def test_to_counter_multi_order_default(self):
        counter = Ngrams(2, min_n=1)
        counter.count(["a", "b", "c"])
        result = counter.to_counter()
        # Default order is the highest (2), so only bigrams
        assert result[("a", "b")] == 1
        assert result[("b", "c")] == 1
        assert len(result) == 2

    def test_to_counter_multi_order_specific(self):
        counter = Ngrams(2, min_n=1)
        counter.count(["a", "b", "c"])
        result = counter.to_counter(order=1)
        assert result[("a",)] == 1
        assert result[("b",)] == 1
        assert result[("c",)] == 1
        assert len(result) == 3

    def test_to_counter_invalid_order_raises(self):
        counter = Ngrams(3, min_n=2)
        with pytest.raises(ValueError, match="order must be between"):
            counter.to_counter(order=1)
