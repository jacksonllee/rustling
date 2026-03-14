# Benchmarks

This directory contains benchmarking scripts to compare Rustling (Rust + PyO3) against pure Python packages. All benchmarks use the HKCanCor (Hong Kong Cantonese Corpus) via [pycantonese](https://pycantonese.org/) as a unified data source.

**GitHub**: https://github.com/jacksonllee/rustling/tree/main/benchmarks

## Directory Structure

```
benchmarks/
├── README.md
├── run_chat.py        # CHAT parsing benchmark (Rustling vs pylangacq)
├── run_hmm.py         # HMM benchmark (Rustling vs hmmlearn)
├── run_lm.py          # Language model benchmark (Rustling vs NLTK)
├── run_wordseg.py     # Word segmentation benchmark (Rustling vs wordseg)
├── run_perceptron_pos_tagger.py  # POS tagger benchmark (Rustling vs NLTK PerceptronTagger)
├── update_docs.py     # Update benchmark table in python/docs/index.rst
└── common/
    ├── __init__.py
    └── data.py        # Shared HKCanCor data loader
```

## Data Source

All benchmarks use the **HKCanCor** corpus (~10K Cantonese sentences with POS tags), loaded via pycantonese. The shared data loader in `common/data.py` converts the corpus into the format each benchmark needs:

- **Tagging**: tagged sentences `[(word, tag), ...]` for training, untagged word lists for testing
- **Word segmentation**: word tuples for training, concatenated strings for testing
- **HMM**: word sequences (tags stripped) for unsupervised Baum-Welch EM training and Viterbi decoding
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
| HMM | [hmmlearn](https://hmmlearn.readthedocs.io/) CategoricalHMM |
| Word Segmentation | [wordseg](https://pypi.org/project/wordseg/) |
| POS Tagging | [NLTK](https://www.nltk.org/) PerceptronTagger |
| Language Models | [NLTK](https://www.nltk.org/) nltk.lm |

All benchmarks degrade gracefully if a comparison library is not installed.

For benchmark results, see the [performance page](https://docs.rustling.io/#performance).

---

## Running Benchmarks

Each script supports `--quick` (fewer iterations), `--export FILE` (JSON output), and `--quiet`:

```bash
python benchmarks/run_chat.py
python benchmarks/run_hmm.py
python benchmarks/run_wordseg.py
python benchmarks/run_perceptron_pos_tagger.py
python benchmarks/run_lm.py
```

## Updating Documentation

After running benchmarks with `--export`, update the performance table in the docs:

```bash
python benchmarks/run_chat.py --export benchmarks/.results/chat.json
python benchmarks/run_hmm.py --export benchmarks/.results/hmm.json
python benchmarks/run_wordseg.py --export benchmarks/.results/wordseg.json
python benchmarks/run_perceptron_pos_tagger.py --export benchmarks/.results/tagger.json
python benchmarks/run_lm.py --export benchmarks/.results/lm.json

python benchmarks/update_docs.py --from-json benchmarks/.results/
```

## Tips

- Use `--release` when building Rustling for accurate benchmarks: `maturin develop --release`
- Close other applications to reduce noise
- Run multiple times to verify consistency
- Use `--quiet` with `--export` for machine-readable output only
