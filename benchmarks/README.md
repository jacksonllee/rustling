# Benchmarks

This directory contains benchmarking scripts to compare `rustling.wordseg` (Rust implementation with Python bindings) against the pure Python `wordseg` package.

## Prerequisites

Install both packages:

```bash
# Install rustling (from this repo)
maturin develop --release

# Install pure Python wordseg for comparison
pip install wordseg
```

## Scripts

### `run_wordseg.py`

Main benchmark comparing both implementations across multiple configurations.

```bash
# Quick sanity check
python benchmarks/run_wordseg.py --quick

# Full benchmark suite
python benchmarks/run_wordseg.py

# Export results to JSON
python benchmarks/run_wordseg.py --export results.json
```

### `realistic.py`

Benchmark with realistic text data (Chinese/English word lists).

```bash
# Chinese text simulation (default)
python benchmarks/realistic.py

# English text simulation
python benchmarks/realistic.py --lang english

# Custom parameters
python benchmarks/realistic.py --sentences 5000 --iterations 10
```

### `scaling.py`

Shows how performance scales with increasing data sizes.

```bash
# Run scaling benchmark
python benchmarks/scaling.py

# With ASCII chart visualization
python benchmarks/scaling.py --plot

# Custom sizes
python benchmarks/scaling.py --sizes 100,500,1000,5000,10000
```

## Example Output

```
============================================================
WORDSEG BENCHMARK: rustling (Rust) vs wordseg (Python)
============================================================

--- Config: 1000 vocab, 1000 sentences ---

📊 LongestStringMatching:
  rustling:
    Total time: 0.0234s (5 iterations)
    Sentences/second: 213,675
  wordseg:
    Total time: 0.4567s (5 iterations)
    Sentences/second: 10,948

  ⚡ Speedup: 19.5x faster
     rustling is 19.5x faster than pure Python wordseg
```

## Tips

- Use `--release` when building rustling for accurate benchmarks: `maturin develop --release`
- Close other applications to reduce noise
- Run multiple times to verify consistency
- The speedup factor depends on data size and complexity
