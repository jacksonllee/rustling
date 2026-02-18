# Benchmarks

This directory contains benchmarking scripts to compare Rustling (Rust + PyO3) against pure Python packages. All benchmarks use the HKCanCor (Hong Kong Cantonese Corpus) via [pycantonese](https://pycantonese.org/) as a unified data source.

**GitHub**: https://github.com/jacksonllee/rustling/tree/main/benchmarks

## Directory Structure

```
benchmarks/
├── README.md
├── run_chat.py        # CHAT parsing benchmark (Rustling vs pylangacq)
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
| CHAT Parsing | [pylangacq](https://pylangacq.org/) |
| Word Segmentation | [wordseg](https://pypi.org/project/wordseg/) |
| POS Tagging | [NLTK](https://www.nltk.org/) PerceptronTagger |
| Language Models | [NLTK](https://www.nltk.org/) nltk.lm |

All benchmarks degrade gracefully if a comparison library is not installed.

---

## CHAT Parsing

Compare `rustling.chat.CHAT` against [pylangacq](https://pylangacq.org/) for parsing CHAT transcription files (CHILDES/TalkBank format).

Benchmarks loading 339 CHAT files from TalkBank's [testchat](https://github.com/TalkBank/testchat) repository via `from_dir`, `from_zip`, `from_files`, and `from_strs`, plus `words()` and `utterances()` extraction. Test data is auto-downloaded to `~/.rustling/testchat/` on first run.

```bash
python benchmarks/run_chat.py
python benchmarks/run_chat.py --quick
python benchmarks/run_chat.py --export results.json
```

Example output (Apple M1 Pro):

```
============================================================
CHAT BENCHMARK: Rustling (Rust) vs pylangacq (Python)
============================================================

from_dir (loading all testchat/good files):
  rustling:
    Total time: 0.0406s (10 iterations)
  pylangacq:
    Total time: 2.2153s (10 iterations)

  Speedup: 54.5x faster

from_zip (loading from ZIP archive):
  rustling:
    Total time: 0.0568s (10 iterations)
  pylangacq:
    Total time: 2.7280s (10 iterations)

  Speedup: 48.0x faster

from_files (loading 339 individual files):
  rustling:
    Total time: 0.0352s (10 iterations)
  pylangacq:
    Total time: 2.2253s (10 iterations)

  Speedup: 63.2x faster

from_strs (parsing 339 in-memory strings):
  rustling:
    Total time: 0.0189s (10 iterations)
  pylangacq:
    Total time: 2.1782s (10 iterations)

  Speedup: 115.5x faster

words() extraction:
  rustling:
    Total time: 0.0019s (10 iterations)
    Detail: 3359 words
  pylangacq:
    Total time: 0.0051s (10 iterations)
    Detail: 3342 words

  Speedup: 2.7x faster

utterances() extraction:
  rustling:
    Total time: 0.0001s (10 iterations)
    Detail: 929 utterances
  pylangacq:
    Total time: 0.0011s (10 iterations)
    Detail: 838 utterances

  Speedup: 15.0x faster
```

Note: The loading benchmarks (`from_dir`, `from_zip`, `from_files`, `from_strs`) are the most significant — they measure parsing 339 CHAT files. `from_strs` isolates pure parsing speed (no I/O). For `words()` and `utterances()` extraction (after data is already loaded), both implementations are fast since they return pre-parsed data.

---

## Word Segmentation

Compare `rustling.wordseg` against the pure Python `wordseg` package.

Benchmarks `LongestStringMatching` and `RandomSegmenter` on HKCanCor data.

```bash
python benchmarks/run_wordseg.py
python benchmarks/run_wordseg.py --quick
python benchmarks/run_wordseg.py --export results.json
```

Example output (Apple M1 Pro):

```
============================================================
WORDSEG BENCHMARK: Rustling (Rust) vs wordseg (Python)
============================================================

📊 LongestStringMatching:
  rustling:
    Total time: 0.0281s (5 iterations)
    Sentences/second: 519,871
  wordseg:
    Total time: 0.2417s (5 iterations)
    Sentences/second: 60,532

  ⚡ Speedup: 8.6x faster

📊 RandomSegmenter:
  rustling:
    Total time: 0.0357s (5 iterations)
    Sentences/second: 410,323
  wordseg:
    Total time: 0.0398s (5 iterations)
    Sentences/second: 367,308

  ⚡ Speedup: 1.1x faster
```

---

## POS Tagging

Compare `rustling.tagging.AveragedPerceptronTagger` against NLTK's `PerceptronTagger` on Cantonese HKCanCor data. Benchmarks both training and tagging speed.

```bash
python benchmarks/run_tagger.py
python benchmarks/run_tagger.py --quick
python benchmarks/run_tagger.py --export results.json
```

Example output (Apple M1 Pro):

```
======================================================================
POS TAGGER BENCHMARK: Rustling (Rust) vs NLTK PerceptronTagger (Python)
======================================================================

--- Training (3 iterations) ---

  rustling.tagging.AveragedPerceptronTagger:
    Training time: 2.4574s

  NLTK PerceptronTagger:
    Training time: 13.3782s

  ⚡ Training speedup: 5.4x faster

--- Tagging (5 iterations) ---

  rustling.tagging.AveragedPerceptronTagger:
    Tagging time: 0.1148s (25,479 sentences/sec)

  NLTK PerceptronTagger:
    Tagging time: 0.8233s (3,554 sentences/sec)

  ⚡ Tagging speedup: 7.2x faster
```

---

## Language Models

Compare `rustling.lm` (MLE, Lidstone, Laplace) against NLTK's `nltk.lm` module. Benchmarks fit (training), score (probability computation), and generate (text generation).

```bash
python benchmarks/run_lm.py
python benchmarks/run_lm.py --quick
python benchmarks/run_lm.py --export results.json
```

Example output (Apple M1 Pro):

```
======================================================================
LANGUAGE MODEL BENCHMARK: Rustling (Rust) vs NLTK (Python)
======================================================================

Config: 11703 sentences, order=3, 5000 score pairs, 500 generate words

📊 MLE:
  [fit]      ⚡ 9.7x faster
  [score]    ⚡ 2.0x faster
  [generate] ⚡ 111.6x faster

📊 Lidstone:
  [fit]      ⚡ 9.7x faster
  [score]    ⚡ 2.3x faster
  [generate] ⚡ 79.9x faster

📊 Laplace:
  [fit]      ⚡ 10.4x faster
  [score]    ⚡ 2.5x faster
  [generate] ⚡ 102.8x faster
```

---

## Tips

- Use `--release` when building Rustling for accurate benchmarks: `maturin develop --release`
- Close other applications to reduce noise
- Run multiple times to verify consistency
- Use `--quiet` with `--export` for machine-readable output only
