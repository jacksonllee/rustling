# Benchmarks

This directory contains benchmarking scripts to compare `rustling` (Rust implementations with Python bindings) against Python packages including wordseg, pycantonese, NLTK, and spaCy.

**GitHub**: https://github.com/jacksonllee/rustling/tree/main/benchmarks

## Directory Structure

```
benchmarks/
├── README.md              # This file
├── common/                # Shared utilities
│   ├── optional_imports.py  # Graceful optional dependency handling
│   └── __init__.py
├── wordseg/               # Word segmentation benchmarks
│   ├── run_wordseg.py     # Main wordseg benchmark
│   ├── realistic.py       # Benchmark with realistic text data
│   └── scaling.py         # Scaling analysis
└── taggers/               # POS tagger benchmarks
    ├── run_tagger.py      # Main tagger benchmark (Cantonese)
    ├── run_tagger_english.py  # English POS tagging (rustling, NLTK, spaCy)
    ├── realistic.py       # Benchmark with HKCanCor corpus
    └── scaling.py         # Scaling analysis
```

## Prerequisites

### Core Dependencies (Required)

```bash
# Install rustling (from this repo)
maturin develop --release

# Or with uv (faster)
uv run maturin develop --release
```

### Comparison Libraries (Optional)

```bash
# Core comparisons (existing)
pip install wordseg==0.0.2 pycantonese==3.4.0

# NLTK (for English POS tagging)
pip install nltk
python -m nltk.downloader punkt averaged_perceptron_tagger universal_tagset brown treebank

# spaCy with English model (for English POS tagging)
pip install spacy
python -m spacy download en_core_web_sm

# Install all with uv (recommended)
uv sync --group benchmarks
```

**Note:** All benchmarks work with graceful degradation. If a library is not installed, its benchmarks will be skipped with a helpful message.

---

## Library Comparison Matrix

Different benchmarks compare different libraries. Here's what each benchmark covers:

| Benchmark Type | rustling | wordseg | pycantonese | NLTK | spaCy |
|----------------|----------|---------|-------------|------|-------|
| **Word Segmentation** (Chinese/Cantonese) | ✓ | ✓ | ✗ | ✗ | ✗ |
| **POS Tagging** (Cantonese) | ✓ | ✗ | ✓ | ✗ | ✗ |
| **POS Tagging** (English) | ✓* | ✗ | ✗ | ✓ | ✓ |

*Can be trained on English data using averaged perceptron tagger

---

## Word Segmentation Benchmarks

Compare `rustling.wordseg` against the pure Python `wordseg` package (v0.0.2).

### `wordseg/run_wordseg.py`

Main benchmark comparing both implementations across multiple configurations.

```bash
# Quick sanity check
python benchmarks/wordseg/run_wordseg.py --quick

# Full benchmark suite
python benchmarks/wordseg/run_wordseg.py

# Export results to JSON
python benchmarks/wordseg/run_wordseg.py --export results.json
```

### `wordseg/realistic.py`

Benchmark with realistic text data (Chinese/English word lists).

```bash
# Chinese text simulation (default)
python benchmarks/wordseg/realistic.py

# English text simulation
python benchmarks/wordseg/realistic.py --lang english

# Custom parameters
python benchmarks/wordseg/realistic.py --sentences 5000 --iterations 10
```

### `wordseg/scaling.py`

Shows how performance scales with increasing data sizes.

```bash
# Run scaling benchmark
python benchmarks/wordseg/scaling.py

# With ASCII chart visualization
python benchmarks/wordseg/scaling.py --plot

# Custom sizes
python benchmarks/wordseg/scaling.py --sizes 100,500,1000,5000,10000
```

---

## POS Tagger Benchmarks (Cantonese)

Compare `rustling.taggers.AveragedPerceptronTagger` against `pycantonese` (v3.4.0)'s `POSTagger` on Cantonese text.

### `taggers/run_tagger.py`

Main benchmark comparing both implementations.

```bash
# Quick sanity check
python benchmarks/taggers/run_tagger.py --quick

# Full benchmark suite
python benchmarks/taggers/run_tagger.py --full

# Export results to JSON
python benchmarks/taggers/run_tagger.py --export results.json
```

### `taggers/realistic.py`

Benchmark with realistic Cantonese corpus data from HKCanCor.

```bash
# Quick benchmark (smaller data subset)
python benchmarks/taggers/realistic.py --quick

# Full benchmark with complete HKCanCor data
python benchmarks/taggers/realistic.py --full
```

### `taggers/scaling.py`

Shows how training and tagging performance scales with data size.

```bash
# Run scaling benchmark
python benchmarks/taggers/scaling.py

# With ASCII chart visualization
python benchmarks/taggers/scaling.py --plot

# Custom sizes
python benchmarks/taggers/scaling.py --sizes 500,1000,2000,5000,10000
```

---

## POS Tagger Benchmarks (English)

Compare `rustling.taggers.AveragedPerceptronTagger`, NLTK `pos_tag()`, and spaCy on English text.

**Note**: This benchmark includes pre-trained models (NLTK, spaCy) which cannot be retrained. Comparisons focus on inference speed rather than training speed or accuracy.

### `taggers/run_tagger_english.py`

English POS tagging benchmark with two modes:

- **Training mode**: Benchmark trainable models (rustling) on Brown corpus
- **Inference mode**: Benchmark all models including pre-trained (NLTK, spaCy, rustling)

```bash
# Quick inference benchmark (default)
python benchmarks/taggers/run_tagger_english.py --quick

# Full inference benchmark
python benchmarks/taggers/run_tagger_english.py --mode inference

# Training mode (rustling only, trainable models)
python benchmarks/taggers/run_tagger_english.py --mode training

# Both modes
python benchmarks/taggers/run_tagger_english.py --mode both

# Use synthetic data instead of Brown corpus
python benchmarks/taggers/run_tagger_english.py --synthetic

# Export results
python benchmarks/taggers/run_tagger_english.py --export results.json
```

**Example Output** (inference/tagging speed comparison):

```
============================================================
INFERENCE MODE: All Models (Pre-trained + Trainable)
============================================================

✓ NLTK loaded successfully
✓ spaCy loaded successfully (model: en_core_web_sm)

Test sentences: 2000

📊 rustling.taggers.AveragedPerceptronTagger:
  Note: Using model trained on synthetic data
  Tagging time: 0.0523s (38,240 sentences/sec)

📊 NLTK pos_tag:
  Note: Pre-trained on Penn Treebank (averaged perceptron)
  Tagging time: 0.2104s (9,506 sentences/sec)

📊 spaCy (en_core_web_sm):
  Note: Pre-trained neural network model
  Tagging time: 1.2341s (1,621 sentences/sec)

============================================================
SPEEDUP COMPARISON
============================================================

Fastest: rustling.AveragedPerceptronTagger (0.0523s)

Relative performance:
  NLTK pos_tag: 4.0x slower
  spaCy en_core_web_sm: 23.6x slower

Note: rustling vs NLTK is apples-to-apples (both averaged perceptron)
      spaCy uses a more complex model for higher accuracy
```

---

## Example Output

### Word Segmentation

```
============================================================
WORDSEG BENCHMARK: rustling (Rust) vs wordseg (Python)
============================================================

--- Config: 1000 vocab, 1000 sentences ---

📊 LongestStringMatching:
  rustling:
    Total time: 0.0056s (5 iterations)
    Sentences/second: 893,928
  wordseg:
    Total time: 0.0560s (5 iterations)
    Sentences/second: 89,210

  ⚡ Speedup: 10.0x faster
```

### POS Tagging (Cantonese)

```
============================================================
Benchmark: HKCanCor Realistic
  Training sentences: 8503
  Test sentences: 8503
============================================================

  rustling.taggers.AveragedPerceptronTagger:
    Tagging time: 0.23s (37,000 sentences/sec)

  pycantonese (v3.4.0) pos_tagging.POSTagger:
    Tagging time: 1.28s (6,600 sentences/sec)

  Speedup (rustling vs pycantonese):
    Tagging: 5.6x faster
```

---

## Tips

- Use `--release` when building rustling for accurate benchmarks: `maturin develop --release`
- Close other applications to reduce noise
- Run multiple times to verify consistency
- The speedup factor depends on data size and complexity
