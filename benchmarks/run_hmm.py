#!/usr/bin/env python
"""Benchmark rustling.hmm vs hmmlearn CategoricalHMM.

Compares unsupervised Baum-Welch EM training, Viterbi decoding, and
Forward-algorithm scoring using HKCanCor corpus data.

Usage:
    python benchmarks/run_hmm.py
    python benchmarks/run_hmm.py --quick
    python benchmarks/run_hmm.py --export results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common.data import hmm_data, load_hkcancor  # noqa: E402


def try_import_rustling_hmm() -> dict[str, Any]:
    """Try to import Rustling's HiddenMarkovModel."""
    try:
        from rustling.hmm import HiddenMarkovModel

        return {"available": True, "class": HiddenMarkovModel}
    except ImportError as e:
        return {"available": False, "error": str(e)}


def try_import_hmmlearn() -> dict[str, Any]:
    """Try to import hmmlearn's CategoricalHMM."""
    try:
        from hmmlearn.hmm import CategoricalHMM

        return {"available": True, "class": CategoricalHMM}
    except ImportError as e:
        return {"available": False, "error": str(e)}


def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    """Build vocabulary mapping from training sequences."""
    vocab: dict[str, int] = {}
    for seq in sequences:
        for word in seq:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def prepare_hmmlearn_data(
    sequences: list[list[str]],
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert string sequences to hmmlearn format.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, lengths) where X is shape (total_obs, 1) int array,
        lengths is shape (n_sequences,) int array.
    """
    oov_id = len(vocab)
    encoded = []
    lengths = []
    for seq in sequences:
        ids = [vocab.get(w, oov_id) for w in seq]
        encoded.extend(ids)
        lengths.append(len(seq))
    X = np.array(encoded, dtype=np.int32).reshape(-1, 1)
    lengths = np.array(lengths, dtype=np.int32)
    return X, lengths


def benchmark_rustling_fit(
    cls: type,
    train_sequences: list[list[str]],
    n_states: int,
    n_iter: int,
    tolerance: float,
    iterations: int,
) -> float:
    """Benchmark Rustling EM training time.

    Returns
    -------
    float
        Average fit time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        model = cls(
            n_states=n_states,
            n_iter=n_iter,
            tolerance=tolerance,
            random_seed=42,
        )
        start = time.perf_counter()
        model.fit(train_sequences)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def benchmark_hmmlearn_fit(
    cls: type,
    train_X: np.ndarray,
    train_lengths: np.ndarray,
    n_states: int,
    n_iter: int,
    tolerance: float,
    n_features: int,
    iterations: int,
) -> float:
    """Benchmark hmmlearn EM training time.

    Returns
    -------
    float
        Average fit time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        model = cls(
            n_components=n_states,
            n_iter=n_iter,
            tol=tolerance,
            n_features=n_features,
            random_state=42,
        )
        start = time.perf_counter()
        model.fit(train_X, train_lengths)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def benchmark_rustling_predict(
    model: Any,
    test_sequences: list[list[str]],
    iterations: int,
) -> float:
    """Benchmark Rustling Viterbi decoding time.

    Returns
    -------
    float
        Average predict time in seconds (excludes warmup iteration).
    """
    # Warmup: avoid cold-start effects
    model.predict(test_sequences)
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        model.predict(test_sequences)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def benchmark_hmmlearn_predict(
    model: Any,
    test_X: np.ndarray,
    test_lengths: np.ndarray,
    iterations: int,
) -> float:
    """Benchmark hmmlearn Viterbi decoding time.

    Returns
    -------
    float
        Average predict time in seconds (excludes warmup iteration).
    """
    # Warmup: avoid cold-start effects
    model.predict(test_X, test_lengths)
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        model.predict(test_X, test_lengths)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def benchmark_rustling_score(
    model: Any,
    test_sequences: list[list[str]],
    iterations: int,
) -> float:
    """Benchmark Rustling Forward-algorithm scoring time.

    Returns
    -------
    float
        Average score time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        model.score(test_sequences)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def benchmark_hmmlearn_score(
    model: Any,
    test_X: np.ndarray,
    test_lengths: np.ndarray,
    iterations: int,
) -> float:
    """Benchmark hmmlearn Forward-algorithm scoring time.

    Returns
    -------
    float
        Average score time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        model.score(test_X, test_lengths)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def run_benchmarks(
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all benchmarks.

    Parameters
    ----------
    quick : bool, default=False
        If True, use smaller data and fewer iterations.
    verbose : bool, default=True
        If True, print results.

    Returns
    -------
    dict[str, Any]
        Benchmark results.
    """
    rustling_info = try_import_rustling_hmm()
    hmmlearn_info = try_import_hmmlearn()

    if verbose:
        if rustling_info["available"]:
            print("✓ rustling.hmm loaded successfully")
        else:
            print(f"✗ rustling.hmm not available: {rustling_info.get('error', '')}")
        if hmmlearn_info["available"]:
            print("✓ hmmlearn CategoricalHMM loaded successfully")
        else:
            print(f"✗ hmmlearn not available: {hmmlearn_info.get('error', '')}")

    if not rustling_info["available"] and not hmmlearn_info["available"]:
        print("\nError: Neither implementation is available.")
        sys.exit(1)

    # Load data
    if verbose:
        print("\nLoading HKCanCor corpus...")
    tagged_sents = load_hkcancor()
    train_sequences, test_sequences = hmm_data(tagged_sents)

    # Shared HMM parameters
    n_states = 10
    n_iter = 100
    tolerance = 1e-6

    if quick:
        train_sequences = train_sequences[:500]
        test_sequences = test_sequences[:100]
        n_iter = 20
        fit_iterations = 2
        predict_iterations = 3
        score_iterations = 3
    else:
        fit_iterations = 3
        predict_iterations = 5
        score_iterations = 5

    if verbose:
        print(f"Training sequences: {len(train_sequences)}")
        print(f"Test sequences: {len(test_sequences)}")
        print(f"Hidden states: {n_states}, EM iterations: {n_iter}")

    # Build vocabulary and prepare hmmlearn data (outside timing)
    vocab = build_vocab(train_sequences)
    n_features = len(vocab) + 1  # +1 for OOV
    train_X, train_lengths = prepare_hmmlearn_data(train_sequences, vocab)
    test_X, test_lengths = prepare_hmmlearn_data(test_sequences, vocab)

    results: dict[str, Any] = {
        "n_states": n_states,
        "n_iter": n_iter,
        "tolerance": tolerance,
        "num_train": len(train_sequences),
        "num_test": len(test_sequences),
        "n_features": int(n_features),
        "benchmarks": {},
    }

    print(
        "\n"
        + "=" * 70
        + "\nHMM BENCHMARK: Rustling (Rust) vs hmmlearn CategoricalHMM"
        + "\n"
        + "=" * 70
    )

    # --- Fit (EM Training) ---
    print(f"\n--- Fit / EM Training ({fit_iterations} iterations) ---")

    rustling_fit_time = None
    if rustling_info["available"]:
        rustling_fit_time = benchmark_rustling_fit(
            rustling_info["class"],
            train_sequences,
            n_states,
            n_iter,
            tolerance,
            fit_iterations,
        )
        if verbose:
            print(
                f"\n  rustling.hmm.HiddenMarkovModel:"
                f"\n    Fit time: {rustling_fit_time:.4f}s"
            )

    hmmlearn_fit_time = None
    if hmmlearn_info["available"]:
        hmmlearn_fit_time = benchmark_hmmlearn_fit(
            hmmlearn_info["class"],
            train_X,
            train_lengths,
            n_states,
            n_iter,
            tolerance,
            n_features,
            fit_iterations,
        )
        if verbose:
            print(
                f"\n  hmmlearn CategoricalHMM:"
                f"\n    Fit time: {hmmlearn_fit_time:.4f}s"
            )

    if rustling_fit_time and hmmlearn_fit_time:
        speedup = hmmlearn_fit_time / rustling_fit_time
        print(f"\n  ⚡ Fit speedup: {speedup:.1f}x faster")
        results["benchmarks"]["fit"] = {
            "rustling": {"time_seconds": rustling_fit_time},
            "hmmlearn": {"time_seconds": hmmlearn_fit_time},
            "speedup": speedup,
        }

    # --- Predict (Viterbi Decoding) ---
    print(f"\n--- Predict / Viterbi Decoding ({predict_iterations} iterations) ---")

    # Pre-train models for predict and score benchmarks
    rustling_model = None
    if rustling_info["available"]:
        rustling_model = rustling_info["class"](
            n_states=n_states,
            n_iter=n_iter,
            tolerance=tolerance,
            random_seed=42,
        )
        rustling_model.fit(train_sequences)

    hmmlearn_model = None
    if hmmlearn_info["available"]:
        hmmlearn_model = hmmlearn_info["class"](
            n_components=n_states,
            n_iter=n_iter,
            tol=tolerance,
            n_features=n_features,
            random_state=42,
        )
        hmmlearn_model.fit(train_X, train_lengths)

    rustling_predict_time = None
    if rustling_model is not None:
        rustling_predict_time = benchmark_rustling_predict(
            rustling_model, test_sequences, predict_iterations
        )
        sps = len(test_sequences) / rustling_predict_time
        if verbose:
            print(
                f"\n  rustling.hmm.HiddenMarkovModel:"
                f"\n    Predict time: {rustling_predict_time:.4f}s"
                f" ({sps:,.0f} sequences/sec)"
            )

    hmmlearn_predict_time = None
    if hmmlearn_model is not None:
        hmmlearn_predict_time = benchmark_hmmlearn_predict(
            hmmlearn_model, test_X, test_lengths, predict_iterations
        )
        sps = len(test_sequences) / hmmlearn_predict_time
        if verbose:
            print(
                f"\n  hmmlearn CategoricalHMM:"
                f"\n    Predict time: {hmmlearn_predict_time:.4f}s"
                f" ({sps:,.0f} sequences/sec)"
            )

    if rustling_predict_time and hmmlearn_predict_time:
        speedup = hmmlearn_predict_time / rustling_predict_time
        print(f"\n  ⚡ Predict speedup: {speedup:.1f}x faster")
        results["benchmarks"]["predict"] = {
            "rustling": {"time_seconds": rustling_predict_time},
            "hmmlearn": {"time_seconds": hmmlearn_predict_time},
            "speedup": speedup,
        }

    # --- Score (Forward Algorithm) ---
    print(f"\n--- Score / Forward Algorithm ({score_iterations} iterations) ---")

    rustling_score_time = None
    if rustling_model is not None:
        rustling_score_time = benchmark_rustling_score(
            rustling_model, test_sequences, score_iterations
        )
        sps = len(test_sequences) / rustling_score_time
        if verbose:
            print(
                f"\n  rustling.hmm.HiddenMarkovModel:"
                f"\n    Score time: {rustling_score_time:.4f}s"
                f" ({sps:,.0f} sequences/sec)"
            )

    hmmlearn_score_time = None
    if hmmlearn_model is not None:
        hmmlearn_score_time = benchmark_hmmlearn_score(
            hmmlearn_model, test_X, test_lengths, score_iterations
        )
        sps = len(test_sequences) / hmmlearn_score_time
        if verbose:
            print(
                f"\n  hmmlearn CategoricalHMM:"
                f"\n    Score time: {hmmlearn_score_time:.4f}s"
                f" ({sps:,.0f} sequences/sec)"
            )

    if rustling_score_time and hmmlearn_score_time:
        speedup = hmmlearn_score_time / rustling_score_time
        print(f"\n  ⚡ Score speedup: {speedup:.1f}x faster")
        results["benchmarks"]["score"] = {
            "rustling": {"time_seconds": rustling_score_time},
            "hmmlearn": {"time_seconds": hmmlearn_score_time},
            "speedup": speedup,
        }

    # Compute speedups summary
    speedups: dict[str, float] = {}
    for op in ["fit", "predict", "score"]:
        bench = results["benchmarks"].get(op, {})
        if "speedup" in bench:
            speedups[op.capitalize()] = bench["speedup"]
    results["speedups"] = speedups

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark rustling.hmm vs hmmlearn CategoricalHMM"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller data and fewer iterations",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (useful with --export)",
    )

    args = parser.parse_args()

    results = run_benchmarks(
        quick=args.quick,
        verbose=not args.quiet,
    )

    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to: {export_path}")


if __name__ == "__main__":
    main()
