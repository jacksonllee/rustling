"""Benchmark regression tests.

These tests run quick benchmarks and verify that Rustling's speedup
over pure Python implementations hasn't regressed catastrophically.
The floors are set conservatively (50% of documented values) to avoid
flakiness across different machines.

Run with:
    uv run maturin develop --release
    uv sync --group benchmarks
    uv run pytest -m benchmark -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make benchmarks/ importable
_BENCHMARKS_DIR = str(Path(__file__).resolve().parent.parent.parent / "benchmarks")
if _BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARKS_DIR)


@pytest.mark.benchmark
class TestLMBenchmark:
    """Language model benchmark regression tests."""

    @pytest.fixture(scope="class")
    def lm_results(self):
        pytest.importorskip("nltk")
        from run_lm import run_benchmarks

        return run_benchmarks(quick=True, verbose=False)

    def test_fit_speedup(self, lm_results):
        """Documented: ~9x. Floor: 4x."""
        speedups = lm_results.get("speedups", {})
        assert "Fit" in speedups, "No fit speedup computed"
        assert (
            speedups["Fit"] >= 4.0
        ), f"Fit speedup {speedups['Fit']:.1f}x below floor 4x"

    def test_score_speedup(self, lm_results):
        """Documented: ~3x. Floor: 1x."""
        speedups = lm_results.get("speedups", {})
        assert "Score" in speedups, "No score speedup computed"
        assert (
            speedups["Score"] >= 1.0
        ), f"Score speedup {speedups['Score']:.1f}x below floor 1x"

    def test_generate_speedup(self, lm_results):
        """Documented: ~32-104x. Floor: 15x."""
        speedups = lm_results.get("speedups", {})
        assert "Generate" in speedups, "No generate speedup computed"
        gen = speedups["Generate"]
        min_speedup = gen["min"] if isinstance(gen, dict) else gen
        assert (
            min_speedup >= 15.0
        ), f"Generate speedup {min_speedup:.1f}x below floor 15x"


@pytest.mark.benchmark
class TestWordsegBenchmark:
    """Word segmentation benchmark regression tests."""

    @pytest.fixture(scope="class")
    def wordseg_results(self):
        pytest.importorskip("wordseg")
        from run_wordseg import run_benchmarks

        return run_benchmarks(quick=True, verbose=False)

    def test_lsm_speedup(self, wordseg_results):
        """Documented: ~7x. Floor: 3x."""
        speedups = wordseg_results.get("speedups", {})
        assert "LongestStringMatching" in speedups, "No LSM speedup computed"
        assert (
            speedups["LongestStringMatching"] >= 3.0
        ), f"LSM speedup {speedups['LongestStringMatching']:.1f}x below floor 3x"


@pytest.mark.benchmark
class TestTaggerBenchmark:
    """POS tagger benchmark regression tests."""

    @pytest.fixture(scope="class")
    def tagger_results(self):
        pytest.importorskip("nltk")
        from run_perceptron_pos_tagger import run_benchmarks

        return run_benchmarks(quick=True, verbose=False)

    def test_training_speedup(self, tagger_results):
        """Documented: ~5x. Floor: 2x."""
        speedups = tagger_results.get("speedups", {})
        assert "Training" in speedups, "No training speedup computed"
        assert (
            speedups["Training"] >= 2.0
        ), f"Training speedup {speedups['Training']:.1f}x below floor 2x"

    def test_tagging_speedup(self, tagger_results):
        """Documented: ~19x. Floor: 5x."""
        speedups = tagger_results.get("speedups", {})
        assert "Tagging" in speedups, "No tagging speedup computed"
        assert (
            speedups["Tagging"] >= 5.0
        ), f"Tagging speedup {speedups['Tagging']:.1f}x below floor 5x"


@pytest.mark.benchmark
class TestHMMBenchmark:
    """HMM benchmark regression tests."""

    @pytest.fixture(scope="class")
    def hmm_results(self):
        pytest.importorskip("hmmlearn")
        from run_hmm import run_benchmarks

        return run_benchmarks(quick=True, verbose=False)

    def test_fit_speedup(self, hmm_results):
        """Documented: ~14x. Floor: 3x."""
        speedups = hmm_results.get("speedups", {})
        assert "Fit" in speedups, "No fit speedup computed"
        assert (
            speedups["Fit"] >= 3.0
        ), f"Fit speedup {speedups['Fit']:.1f}x below floor 3x"

    def test_predict_speedup(self, hmm_results):
        """Documented: ~0.8x (full) / ~2x (quick). Floor: 0.5x."""
        speedups = hmm_results.get("speedups", {})
        assert "Predict" in speedups, "No predict speedup computed"
        assert (
            speedups["Predict"] >= 0.5
        ), f"Predict speedup {speedups['Predict']:.1f}x below floor 0.5x"

    def test_score_speedup(self, hmm_results):
        """Documented: ~5x. Floor: 1x."""
        speedups = hmm_results.get("speedups", {})
        assert "Score" in speedups, "No score speedup computed"
        assert (
            speedups["Score"] >= 1.0
        ), f"Score speedup {speedups['Score']:.1f}x below floor 1x"


@pytest.mark.benchmark
class TestChatBenchmark:
    """CHAT parsing benchmark regression tests."""

    @pytest.fixture(scope="class")
    def chat_results(self):
        pytest.importorskip("pylangacq")
        from run_chat import run_benchmarks

        return run_benchmarks(quick=True, verbose=False)

    def test_from_zip_speedup(self, chat_results):
        """Documented: ~44x. Floor: 10x."""
        speedups = chat_results.get("speedups", {})
        assert "from_zip" in speedups, "No from_zip speedup computed"
        assert (
            speedups["from_zip"] >= 10.0
        ), f"from_zip speedup {speedups['from_zip']:.1f}x below floor 10x"

    def test_from_strs_speedup(self, chat_results):
        """Documented: ~70x. Floor: 20x."""
        speedups = chat_results.get("speedups", {})
        assert "from_strs" in speedups, "No from_strs speedup computed"
        assert (
            speedups["from_strs"] >= 20.0
        ), f"from_strs speedup {speedups['from_strs']:.1f}x below floor 20x"

    def test_utterances_speedup(self, chat_results):
        """Documented: ~14x. Floor: 5x."""
        speedups = chat_results.get("speedups", {})
        assert "utterances()" in speedups, "No utterances() speedup computed"
        assert (
            speedups["utterances()"] >= 5.0
        ), f"utterances() speedup {speedups['utterances()']:.1f}x below floor 5x"

    def test_tokens_speedup(self, chat_results):
        """Documented: ~9x. Floor: 3x."""
        speedups = chat_results.get("speedups", {})
        assert "tokens()" in speedups, "No tokens() speedup computed"
        assert (
            speedups["tokens()"] >= 3.0
        ), f"tokens() speedup {speedups['tokens()']:.1f}x below floor 3x"
