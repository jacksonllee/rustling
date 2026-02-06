# Building the Documentation

This directory contains the Sphinx documentation for rustling.

## Build Instructions

From this directory (`python/docs/`), run:

```bash
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

## Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv pip install -r requirements.txt
```

You'll also need to have rustling installed in your environment (use `maturin develop` from the repo root).

## Other Build Targets

- `make clean` - Remove built documentation
- `make help` - Show all available targets
