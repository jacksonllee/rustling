#!/usr/bin/env python
"""Benchmark rustling.elan vs pympi-ling for ELAN parsing.

Compares ELAN (.eaf) file loading and data extraction speed using CantoMap data.

Usage:
    python benchmarks/run_elan.py
    python benchmarks/run_elan.py --quick
    python benchmarks/run_elan.py --export results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

CANTOMAP_DIR = Path.home() / ".rustling" / "cantomap"


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


def ensure_cantomap() -> Path:
    """Download CantoMap data if not present."""
    if not CANTOMAP_DIR.exists():
        print("Downloading CantoMap data...")
        CANTOMAP_DIR.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # Skip Git LFS (audio files not needed for ELAN parsing)
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/gwinterstein/CantoMap.git",
                str(CANTOMAP_DIR),
            ],
            check=True,
            env=env,
        )
    return CANTOMAP_DIR


def collect_eaf_files(cantomap_dir: Path) -> list[Path]:
    """Collect all .eaf files from CantoMap."""
    eaf_files = sorted(cantomap_dir.rglob("*.eaf"))
    if not eaf_files:
        print("Error: No .eaf files found in CantoMap data.")
        sys.exit(1)
    return eaf_files


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
    rustling_result: BenchmarkResult, pympi_result: BenchmarkResult
) -> None:
    """Print comparison between rustling and pympi-ling."""
    if rustling_result.time_seconds > 0:
        speedup = pympi_result.time_seconds / rustling_result.time_seconds
    else:
        speedup = float("inf")
    print(f"\n  Speedup: {speedup:.1f}x faster")


def run_benchmarks(
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all benchmarks."""
    cantomap_dir = ensure_cantomap()
    eaf_files = collect_eaf_files(cantomap_dir)
    eaf_paths_str = [str(p) for p in eaf_files]
    single_path = eaf_paths_str[0]

    rustling_elan = None
    pympi_eaf = None

    try:
        from rustling.elan import ELAN

        rustling_elan = ELAN
        if verbose:
            print("rustling.elan loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"rustling.elan not available: {e}")

    try:
        from pympi import Elan as pympi_elan_module

        pympi_eaf = pympi_elan_module.Eaf
        if verbose:
            print("pympi-ling loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"pympi-ling not available: {e}")

    if rustling_elan is None and pympi_eaf is None:
        print("\nError: Neither implementation is available.")
        sys.exit(1)

    iterations = 3 if quick else 10

    all_results: dict[str, Any] = {"benchmarks": {}}

    print("\n" + "=" * 60)
    print("ELAN BENCHMARK: Rustling (Rust) vs pympi-ling (Python)")
    print(f"Dataset: CantoMap ({len(eaf_files)} .eaf files)")
    print("=" * 60)

    # Benchmark 1: Parse single EAF file
    if verbose:
        print(f"\nParse single file ({Path(single_path).name}):")

    results = []
    for name, impl in [("rustling", rustling_elan), ("pympi-ling", pympi_eaf)]:
        if impl is None:
            results.append(None)
            continue

        if name == "rustling":

            def parse_single() -> None:
                impl.from_files([single_path])

        else:

            def parse_single() -> None:
                impl(file_path=single_path)

        total_time = time_function(parse_single, iterations=iterations)
        result = BenchmarkResult(
            name="parse_single",
            implementation=name,
            time_seconds=total_time,
            iterations=iterations,
        )
        results.append(result)
        if verbose:
            print_result(result)

    if results[0] and results[1] and verbose:
        print_comparison(results[0], results[1])

    all_results["benchmarks"]["parse_single"] = {
        "rustling": results[0].__dict__ if results[0] else None,
        "pympi-ling": results[1].__dict__ if results[1] else None,
    }

    # Benchmark 2: Parse all EAF files
    if verbose:
        print(f"\nParse all files ({len(eaf_files)} files):")

    results = []
    for name, impl in [("rustling", rustling_elan), ("pympi-ling", pympi_eaf)]:
        if impl is None:
            results.append(None)
            continue

        if name == "rustling":

            def parse_all() -> None:
                impl.from_files(eaf_paths_str, parallel=True)

        else:

            def parse_all() -> None:
                for p in eaf_paths_str:
                    impl(file_path=p)

        total_time = time_function(parse_all, iterations=iterations)
        result = BenchmarkResult(
            name="parse_all",
            implementation=name,
            time_seconds=total_time,
            iterations=iterations,
            detail=f"{len(eaf_files)} files",
        )
        results.append(result)
        if verbose:
            print_result(result)

    if results[0] and results[1] and verbose:
        print_comparison(results[0], results[1])

    all_results["benchmarks"]["parse_all"] = {
        "rustling": results[0].__dict__ if results[0] else None,
        "pympi-ling": results[1].__dict__ if results[1] else None,
    }

    # Compute speedups for each task
    speedups: dict[str, float] = {}
    for task in ["parse_single", "parse_all"]:
        bench = all_results["benchmarks"].get(task, {})
        r = bench.get("rustling")
        p = bench.get("pympi-ling")
        if r and p and r["time_seconds"] > 0:
            speedups[task] = p["time_seconds"] / r["time_seconds"]
    all_results["speedups"] = speedups

    return all_results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark rustling.elan vs pympi-ling for ELAN parsing"
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
