# Benchmarks

This directory contains benchmarking scripts to compare Rustling (Rust + PyO3) against other Python packages with similar functionalities.

**GitHub**: https://github.com/jacksonllee/rustling/tree/main/benchmarks

## Directory Structure

```
benchmarks/
├── README.md
├── run_chat.py        # CHAT parsing benchmark (Rustling vs pylangacq)
├── run_conllu.py      # CoNLL-U parsing benchmark (Rustling vs conllu)
├── run_elan.py        # ELAN parsing benchmark (Rustling vs pympi-ling)
├── run_textgrid.py    # TextGrid parsing benchmark (Rustling vs pympi-ling)
├── run_hmm.py         # HMM benchmark (Rustling vs hmmlearn)
├── run_lm.py          # Language model benchmark (Rustling vs NLTK)
├── run_wordseg.py     # Word segmentation benchmark (Rustling vs wordseg)
├── run_perceptron_pos_tagger.py  # POS tagger benchmark (Rustling vs NLTK PerceptronTagger)
├── update_readme.py   # Update benchmark tables in README files
└── common/
    ├── __init__.py
    └── data.py        # Shared HKCanCor data loader
```

## Data Sources

Most benchmarks use the **HKCanCor** corpus (~10K Cantonese sentences with POS tags), loaded via pycantonese. The shared data loader in `common/data.py` converts the corpus into the format each benchmark needs:

- **Tagging**: tagged sentences `[(word, tag), ...]` for training, untagged word lists for testing
- **Word segmentation**: word tuples for training, concatenated strings for testing
- **HMM**: word sequences (tags stripped) for unsupervised Baum-Welch EM training and Viterbi decoding
- **Language models**: word sequences (tags stripped)

The CoNLL-U benchmark uses the **UD_English-EWT** treebank (English Universal Dependencies data), auto-downloaded to `~/.rustling/ud-english-ewt/`.

The ELAN benchmark uses the **CantoMap** corpus (Cantonese conversation data with ELAN annotations), auto-downloaded to `~/.rustling/cantomap/`.

The TextGrid benchmark uses TextGrid files generated from the CantoMap ELAN data via `rustling.elan.ELAN.to_textgrid_files()`, cached at `~/.rustling/cantomap_textgrid/`.

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
| CoNLL-U Parsing | [conllu](https://github.com/EmilStenstrom/conllu/) |
| ELAN Parsing | [pympi-ling](https://pypi.org/project/pympi-ling/) Eaf |
| TextGrid Parsing | [pympi-ling](https://pypi.org/project/pympi-ling/) TextGrid |
| HMM | [hmmlearn](https://hmmlearn.readthedocs.io/) CategoricalHMM |
| Word Segmentation | [wordseg](https://pypi.org/project/wordseg/) |
| POS Tagging | [NLTK](https://www.nltk.org/) PerceptronTagger |
| Language Models | [NLTK](https://www.nltk.org/) nltk.lm |

All benchmarks degrade gracefully if a comparison library is not installed.

## Results

Benchmarked against Python implementations from NLTK, wordseg (v0.0.5),
pylangacq (v0.19.1), hmmlearn (v0.3.3), pympi-ling (v1.70.2), and conllu (v6.0.0).

| Component | Task | Speedup | vs. |
|---|---|---|---|
| **Language Models** | Fit | **11x** | NLTK |
|  | Score | **2x** | NLTK |
|  | Generate | **86--107x** | NLTK |
| **Word Segmentation** | LongestStringMatching | **9x** | wordseg |
| **POS Tagging** | Training | **5x** | NLTK |
|  | Tagging | **17x** | NLTK |
| **HMM** | Fit | **14x** | hmmlearn |
|  | Predict | **0.9x** | hmmlearn |
|  | Score | **5x** | hmmlearn |
| **CHAT Parsing** | Reading from a ZIP archive | **30x** | pylangacq |
|  | Reading from strings | **35x** | pylangacq |
|  | Parsing utterances | **15x** | pylangacq |
|  | Parsing tokens | **8x** | pylangacq |
| **ELAN Parsing** | Parse single file | **4x** | pympi-ling |
|  | Parse all files | **17x** | pympi-ling |
| **TextGrid Parsing** | Parse single file | **3x** | pympi-ling |
|  | Parse all files | **8x** | pympi-ling |
| **CoNLL-U Parsing** | Parse from strings | **15x** | conllu |
|  | Parse from files | **15x** | conllu |

---

## Running Benchmarks

Each script supports `--quick` (fewer iterations), `--export FILE` (JSON output), and `--quiet`:

```bash
python benchmarks/run_chat.py
python benchmarks/run_conllu.py
python benchmarks/run_elan.py
python benchmarks/run_textgrid.py
python benchmarks/run_hmm.py
python benchmarks/run_wordseg.py
python benchmarks/run_perceptron_pos_tagger.py
python benchmarks/run_lm.py
```

## Updating Benchmark Tables

After running benchmarks with `--export`, update the performance table in `benchmarks/README.md`:

```bash
python benchmarks/run_chat.py --export benchmarks/.results/chat.json
python benchmarks/run_conllu.py --export benchmarks/.results/conllu.json
python benchmarks/run_elan.py --export benchmarks/.results/elan.json
python benchmarks/run_textgrid.py --export benchmarks/.results/textgrid.json
python benchmarks/run_hmm.py --export benchmarks/.results/hmm.json
python benchmarks/run_wordseg.py --export benchmarks/.results/wordseg.json
python benchmarks/run_perceptron_pos_tagger.py --export benchmarks/.results/tagger.json
python benchmarks/run_lm.py --export benchmarks/.results/lm.json

python benchmarks/update_readme.py --from-json benchmarks/.results/
```

## Tips

- Use `--release` when building Rustling for accurate benchmarks: `maturin develop --release`
- Close other applications to reduce noise
- Run multiple times to verify consistency
- Use `--quiet` with `--export` for machine-readable output only
