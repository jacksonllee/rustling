# Benchmarks

This directory contains benchmarking scripts to compare Rustling (Rust + PyO3) against pure Python packages. All benchmarks use the HKCanCor (Hong Kong Cantonese Corpus) via [pycantonese](https://pycantonese.org/) as a unified data source.

**GitHub**: https://github.com/jacksonllee/rustling/tree/main/benchmarks

## Directory Structure

```
benchmarks/
├── README.md
├── run_lm.py          # Language model benchmark (Rustling vs NLTK)
├── run_wordseg.py     # Word segmentation benchmark (Rustling vs wordseg)
├── run_tagger.py      # POS tagger benchmark (Rustling vs NLTK PerceptronTagger)
└── common/
    ├── __init__.py
    └── data.py        # Shared HKCanCor data loader
```

## Data Source

All benchmarks use the **HKCanCor** corpus (~10K Cantonese sentences with POS tags), loaded via pycantonese. The shared data loader in `common/data.py` converts the corpus into the format each benchmark needs:

- **Tagging**: tagged sentences `[(word, tag), ...]` for training, untagged word lists for testing
- **Word segmentation**: word tuples for training, concatenated strings for testing
- **Language models**: word sequences (tags stripped)

## Prerequisites

```bash
# Build Rustling (from repo root)
uv run maturin develop --release

# Install benchmark dependencies
uv sync --group benchmarks
```

### Comparison Libraries

| Benchmark | Comparison Library |
|-----------|--------------------|
| Word Segmentation | [wordseg](https://pypi.org/project/wordseg/) 0.0.2 |
| POS Tagging | [NLTK](https://www.nltk.org/) PerceptronTagger |
| Language Models | [NLTK](https://www.nltk.org/) nltk.lm |

All benchmarks degrade gracefully if a comparison library is not installed.

---

## Word Segmentation

Compare `rustling.wordseg` against the pure Python `wordseg` package.

Benchmarks `LongestStringMatching` and `RandomSegmenter` on HKCanCor data.

```bash
python benchmarks/run_wordseg.py --quick
python benchmarks/run_wordseg.py
python benchmarks/run_wordseg.py --export results.json
```

Example output (`--quick`, Apple M1 Pro):

```
============================================================
WORDSEG BENCHMARK: Rustling (Rust) vs wordseg (Python)
============================================================

📊 LongestStringMatching:
  rustling:
    Total time: 0.0007s (3 iterations)
    Sentences/second: 420,487
  wordseg:
    Total time: 0.0102s (3 iterations)
    Sentences/second: 29,536

  ⚡ Speedup: 14.2x faster

📊 RandomSegmenter:
  rustling:
    Total time: 0.0005s (3 iterations)
    Sentences/second: 601,304
  wordseg:
    Total time: 0.0061s (3 iterations)
    Sentences/second: 48,966

  ⚡ Speedup: 12.3x faster
```

---

## POS Tagging

Compare `rustling.tagging.AveragedPerceptronTagger` against NLTK's `PerceptronTagger` on Cantonese HKCanCor data. Benchmarks both training and tagging speed.

```bash
python benchmarks/run_tagger.py --quick
python benchmarks/run_tagger.py
python benchmarks/run_tagger.py --export results.json
```

Example output (`--quick`, Apple M1 Pro):

```
======================================================================
POS TAGGER BENCHMARK: Rustling (Rust) vs NLTK PerceptronTagger (Python)
======================================================================

--- Training (2 iterations) ---

  rustling.tagging.AveragedPerceptronTagger:
    Training time: 0.1539s

  NLTK PerceptronTagger:
    Training time: 0.7973s

  ⚡ Training speedup: 5.2x faster

--- Tagging (3 iterations) ---

  rustling.tagging.AveragedPerceptronTagger:
    Tagging time: 0.0054s (37,379 sentences/sec)

  NLTK PerceptronTagger:
    Tagging time: 0.0329s (6,073 sentences/sec)

  ⚡ Tagging speedup: 6.2x faster
```

---

## Language Models

Compare `rustling.lm` (MLE, Lidstone, Laplace) against NLTK's `nltk.lm` module. Benchmarks fit (training), score (probability computation), and generate (text generation).

```bash
python benchmarks/run_lm.py --quick
python benchmarks/run_lm.py
python benchmarks/run_lm.py --export results.json
```

Example output (`--quick`, Apple M1 Pro):

```
======================================================================
LANGUAGE MODEL BENCHMARK: Rustling (Rust) vs NLTK (Python)
======================================================================

Config: 500 sentences, order=3, 1000 score pairs, 100 generate words

📊 MLE:
  [fit]      ⚡ 11.5x faster
  [score]    ⚡ 2.0x faster
  [generate] ⚡ 24.7x faster

📊 Lidstone:
  [fit]      ⚡ 11.4x faster
  [score]    ⚡ 2.2x faster
  [generate] ⚡ 34.5x faster

📊 Laplace:
  [fit]      ⚡ 11.7x faster
  [score]    ⚡ 2.3x faster
  [generate] ⚡ 39.3x faster
```

---

## Tips

- Use `--release` when building Rustling for accurate benchmarks: `maturin develop --release`
- Close other applications to reduce noise
- Run multiple times to verify consistency
- Use `--quiet` with `--export` for machine-readable output only
