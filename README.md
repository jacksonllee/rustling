<div align="center">
  <a href="https://github.com/jacksonllee/rustling">
    <img src="https://raw.githubusercontent.com/jacksonllee/rustling/main/python/docs/_static/logo-with-text.svg" alt="Rustling" height="120">
  </a>
</div>
<br>

[![PyPI](https://img.shields.io/pypi/v/rustling.svg)](https://pypi.org/project/rustling/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/rustling.svg)](https://anaconda.org/conda-forge/rustling)
[![crates.io](https://img.shields.io/crates/v/rustling.svg)](https://crates.io/crates/rustling)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19140734.svg)](https://doi.org/10.5281/zenodo.19140734)

Rustling is a blazingly fast library for computational linguistics.
It aims to provide flexible and efficient tools to facilitate further research.

Documentation: [Python](https://docs.rustling.io/) | [Rust](https://docs.rs/rustling)

Currently implemented features:

* Sequence modeling:

   - N-grams and related language models
   - Hidden Markov model
   - Word segmentation
   - Averaged perceptron part-of-speech tagging

* Handling richly formatted data,
  supporting cross-format conversion as well as both local and remote sources for data ingestion:

   - CHAT for TalkBank and CHILDES
   - ELAN for annotated multimedia data
   - TextGrid for Praat annotations
   - CoNLL-U for University Dependencies
   - SRT for SubRip subtitles

## Performance

Rustling is highly performant because it is implemented in Rust under the hood.
For benchmarks comparing Rustling against other Python packages with similar functionalities,
please see [`benchmarks`](https://github.com/jacksonllee/rustling/tree/main/benchmarks).


## Installation

### Python

Using pip:

```bash
pip install rustling
```

Using conda:

```bash
conda install -c conda-forge rustling
```

For Pyodide, pre-built WASM wheels (with multithreading disabled, as Pyodide does not support it)
are available from each [GitHub release](https://github.com/jacksonllee/rustling/releases)
— look for the ``.whl`` file with ``emscripten`` in the filename.

### Rust

```bash
cargo add rustling
```

## License

MIT License
