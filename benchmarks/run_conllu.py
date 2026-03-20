#!/usr/bin/env python
"""Benchmark rustling.conllu vs conllu for CoNLL-U parsing.

Compares CoNLL-U file loading speed using UD_English-EWT treebank data.

Usage:
    python benchmarks/run_conllu.py
    python benchmarks/run_conllu.py --quick
    python benchmarks/run_conllu.py --export results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

UD_ENGLISH_EWT_DIR = Path.home() / ".rustling" / "ud-english-ewt"


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


def ensure_ud_english_ewt() -> Path:
    """Download UD_English-EWT data if not present."""
    if not UD_ENGLISH_EWT_DIR.exists():
        print("Downloading UD_English-EWT data...")
        UD_ENGLISH_EWT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/UniversalDependencies/UD_English-EWT.git",
                str(UD_ENGLISH_EWT_DIR),
            ],
            check=True,
        )
    return UD_ENGLISH_EWT_DIR


def collect_conllu_files(ud_dir: Path) -> list[Path]:
    """Collect all .conllu files from the treebank directory."""
    conllu_files = sorted(ud_dir.glob("*.conllu"))
    if not conllu_files:
        print("Error: No .conllu files found in UD_English-EWT data.")
        sys.exit(1)
    return conllu_files


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
    """Print comparison between rustling and conllu."""
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
    ud_dir = ensure_ud_english_ewt()
    conllu_files = collect_conllu_files(ud_dir)
    conllu_paths_str = [str(p) for p in conllu_files]

    rustling_conllu = None
    conllu_pkg = None

    try:
        from rustling.conllu import CoNLLU

        rustling_conllu = CoNLLU
        if verbose:
            print("rustling.conllu loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"rustling.conllu not available: {e}")

    try:
        import conllu as _conllu_pkg

        conllu_pkg = _conllu_pkg
        if verbose:
            print("conllu loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"conllu not available: {e}")

    if rustling_conllu is None and conllu_pkg is None:
        print("\nError: Neither implementation is available.")
        sys.exit(1)

    iterations = 3 if quick else 10

    all_results: dict[str, Any] = {"benchmarks": {}}

    # Pre-read file contents for from_strs benchmark
    conllu_strs = []
    for f in conllu_files:
        conllu_strs.append(f.read_text(encoding="utf-8"))

    print("\n" + "=" * 60)
    print("CONLLU BENCHMARK: Rustling (Rust) vs conllu (Python)")
    print(f"Dataset: UD_English-EWT ({len(conllu_files)} .conllu files)")
    print("=" * 60)

    # Benchmark 1: from_strs (parse from in-memory strings)
    if verbose:
        print(f"\nfrom_strs (parsing {len(conllu_strs)} in-memory strings):")

    results = []
    for name, impl in [("rustling", rustling_conllu), ("conllu", conllu_pkg)]:
        if impl is None:
            results.append(None)
            continue

        if name == "rustling":

            def parse_strs() -> None:
                impl.from_strs(conllu_strs)

        else:

            def parse_strs() -> None:
                for s in conllu_strs:
                    impl.parse(s)

        total_time = time_function(parse_strs, iterations=iterations)
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
        "conllu": results[1].__dict__ if results[1] else None,
    }

    # Benchmark 2: from_files (parse from disk)
    if verbose:
        print(f"\nfrom_files (parsing {len(conllu_files)} files from disk):")

    results = []
    for name, impl in [("rustling", rustling_conllu), ("conllu", conllu_pkg)]:
        if impl is None:
            results.append(None)
            continue

        if name == "rustling":

            def parse_files() -> None:
                impl.from_files(conllu_paths_str)

        else:

            def parse_files() -> None:
                for p in conllu_paths_str:
                    with open(p, encoding="utf-8") as f:
                        impl.parse(f.read())

        total_time = time_function(parse_files, iterations=iterations)
        result = BenchmarkResult(
            name="from_files",
            implementation=name,
            time_seconds=total_time,
            iterations=iterations,
            detail=f"{len(conllu_files)} files",
        )
        results.append(result)
        if verbose:
            print_result(result)

    if results[0] and results[1] and verbose:
        print_comparison(results[0], results[1])

    all_results["benchmarks"]["from_files"] = {
        "rustling": results[0].__dict__ if results[0] else None,
        "conllu": results[1].__dict__ if results[1] else None,
    }

    # Compute speedups for each task
    speedups: dict[str, float] = {}
    for task in ["from_strs", "from_files"]:
        bench = all_results["benchmarks"].get(task, {})
        r = bench.get("rustling")
        p = bench.get("conllu")
        if r and p and r["time_seconds"] > 0:
            speedups[task] = p["time_seconds"] / r["time_seconds"]
    all_results["speedups"] = speedups

    return all_results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark rustling.conllu vs conllu for CoNLL-U parsing"
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
