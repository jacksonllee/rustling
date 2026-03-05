#!/usr/bin/env python
"""Benchmark rustling.chat vs pylangacq for CHAT parsing.

Compares CHAT file loading and data extraction speed using TalkBank testchat data.

Usage:
    python benchmarks/run_chat.py
    python benchmarks/run_chat.py --quick
    python benchmarks/run_chat.py --export results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TESTCHAT_DIR = Path.home() / ".rustling" / "testchat"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    implementation: str
    time_seconds: float
    iterations: int
    detail: str = ""
    ops_per_second: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.time_seconds > 0:
            self.ops_per_second = self.iterations / self.time_seconds


def ensure_testchat() -> Path:
    """Download TalkBank testchat data if not present."""
    good_dir = TESTCHAT_DIR / "good"
    if not good_dir.exists():
        print("Downloading TalkBank testchat data...")
        TESTCHAT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/TalkBank/testchat.git",
                str(TESTCHAT_DIR),
            ],
            check=True,
        )
    return good_dir


def ensure_testchat_zip(good_dir: Path) -> Path:
    """Create a ZIP of testchat/good/*.cha if not present."""
    zip_path = TESTCHAT_DIR / "good.zip"
    if not zip_path.exists():
        cha_files = sorted(good_dir.glob("*.cha"))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in cha_files:
                zf.write(f, f.name)
    return zip_path


def time_function(
    func: Callable[[], Any],
    iterations: int = 5,
    warmup: int = 1,
) -> float:
    """Time a function over multiple iterations.

    Returns
    -------
    float
        Total time in seconds.
    """
    for _ in range(warmup):
        func()

    gc.collect()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return sum(times)


def print_result(result: BenchmarkResult) -> None:
    """Print a benchmark result."""
    print(f"  {result.implementation}:")
    print(
        f"    Total time: {result.time_seconds:.4f}s ({result.iterations} iterations)"
    )
    if result.detail:
        print(f"    Detail: {result.detail}")


def print_comparison(
    rustling_result: BenchmarkResult, python_result: BenchmarkResult
) -> None:
    """Print comparison between rustling and pylangacq."""
    if rustling_result.time_seconds > 0:
        speedup = python_result.time_seconds / rustling_result.time_seconds
    else:
        speedup = float("inf")
    print(f"\n  Speedup: {speedup:.1f}x faster")


def run_benchmarks(
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all benchmarks."""
    good_dir = ensure_testchat()
    zip_path = ensure_testchat_zip(good_dir)
    zip_path_str = str(zip_path)

    rustling_chat = None
    pylangacq_reader = None

    try:
        from rustling.chat import CHAT

        rustling_chat = CHAT
        if verbose:
            print("rustling.chat loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"rustling.chat not available: {e}")

    try:
        import pylangacq

        pylangacq_reader = pylangacq.Reader
        if verbose:
            print("pylangacq loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"pylangacq not available: {e}")

    if rustling_chat is None and pylangacq_reader is None:
        print("\nError: Neither implementation is available.")
        sys.exit(1)

    iterations = 3 if quick else 10

    all_results: dict[str, Any] = {"benchmarks": {}}

    print("\n" + "=" * 60)
    print("CHAT BENCHMARK: Rustling (Rust) vs pylangacq (Python)")
    print("=" * 60)

    # Benchmark 1: from_zip (loading from ZIP archive)
    if verbose:
        print("\nfrom_zip (loading from ZIP archive):")

    results = []
    for name, impl_cls in [
        ("rustling", rustling_chat),
        ("pylangacq", pylangacq_reader),
    ]:
        if impl_cls is None:
            results.append(None)
            continue

        def load_zip() -> None:
            impl_cls.from_zip(zip_path_str, strict=False)

        total_time = time_function(load_zip, iterations=iterations)
        result = BenchmarkResult(
            name="from_zip",
            implementation=name,
            time_seconds=total_time,
            iterations=iterations,
        )
        results.append(result)
        if verbose:
            print_result(result)

    if results[0] and results[1] and verbose:
        print_comparison(results[0], results[1])

    all_results["benchmarks"]["from_zip"] = {
        "rustling": results[0].__dict__ if results[0] else None,
        "pylangacq": results[1].__dict__ if results[1] else None,
    }

    # Benchmark 2: from_strs (loading from in-memory strings)
    chat_strs = []
    for f in sorted(good_dir.glob("*.cha")):
        chat_strs.append(f.read_text(encoding="utf-8"))

    if verbose:
        print(f"\nfrom_strs (parsing {len(chat_strs)} in-memory strings):")

    results = []
    for name, impl_cls in [
        ("rustling", rustling_chat),
        ("pylangacq", pylangacq_reader),
    ]:
        if impl_cls is None:
            results.append(None)
            continue

        def load_strs() -> None:
            impl_cls.from_strs(chat_strs, strict=False)

        total_time = time_function(load_strs, iterations=iterations)
        result = BenchmarkResult(
            name="from_strs",
            implementation=name,
            time_seconds=total_time,
            iterations=iterations,
        )
        results.append(result)
        if verbose:
            print_result(result)

    if results[0] and results[1] and verbose:
        print_comparison(results[0], results[1])

    all_results["benchmarks"]["from_strs"] = {
        "rustling": results[0].__dict__ if results[0] else None,
        "pylangacq": results[1].__dict__ if results[1] else None,
    }

    # Benchmark 3: utterances() extraction
    if verbose:
        print("\nutterances() extraction:")

    results = []
    for name, impl_cls in [
        ("rustling", rustling_chat),
        ("pylangacq", pylangacq_reader),
    ]:
        if impl_cls is None:
            results.append(None)
            continue

        reader = impl_cls.from_zip(zip_path_str, strict=False)

        def extract_utts() -> None:
            reader.utterances()

        total_time = time_function(extract_utts, iterations=iterations)
        n_utts = len(reader.utterances())
        result = BenchmarkResult(
            name="utterances",
            implementation=name,
            time_seconds=total_time,
            iterations=iterations,
            detail=f"{n_utts} utterances",
        )
        results.append(result)
        if verbose:
            print_result(result)

    if results[0] and results[1] and verbose:
        print_comparison(results[0], results[1])

    all_results["benchmarks"]["utterances"] = {
        "rustling": results[0].__dict__ if results[0] else None,
        "pylangacq": results[1].__dict__ if results[1] else None,
    }

    # Benchmark 4: tokens() extraction
    if verbose:
        print("\ntokens() extraction:")

    results = []
    for name, impl_cls in [
        ("rustling", rustling_chat),
        ("pylangacq", pylangacq_reader),
    ]:
        if impl_cls is None:
            results.append(None)
            continue

        reader = impl_cls.from_zip(zip_path_str, strict=False)

        def extract_tokens() -> None:
            reader.tokens()

        total_time = time_function(extract_tokens, iterations=iterations)
        n_tokens = len(reader.tokens())
        result = BenchmarkResult(
            name="tokens",
            implementation=name,
            time_seconds=total_time,
            iterations=iterations,
            detail=f"{n_tokens} tokens",
        )
        results.append(result)
        if verbose:
            print_result(result)

    if results[0] and results[1] and verbose:
        print_comparison(results[0], results[1])

    all_results["benchmarks"]["tokens"] = {
        "rustling": results[0].__dict__ if results[0] else None,
        "pylangacq": results[1].__dict__ if results[1] else None,
    }

    # Compute speedups for each task
    speedups: dict[str, float] = {}
    for task in ["from_zip", "from_strs", "utterances", "tokens"]:
        bench = all_results["benchmarks"].get(task, {})
        r = bench.get("rustling")
        p = bench.get("pylangacq")
        if r and p and r["time_seconds"] > 0:
            display = f"{task}()" if task in ("utterances", "tokens") else task
            speedups[display] = p["time_seconds"] / r["time_seconds"]
    all_results["speedups"] = speedups

    return all_results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark rustling.chat vs pylangacq for CHAT parsing"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with fewer iterations",
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
        with open(export_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to: {export_path}")


if __name__ == "__main__":
    main()
