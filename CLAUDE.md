# CLAUDE.md

This file provides guidance for Claude when working with this repository.

## Project Overview

**rustling** is a computational linguistics library implemented in Rust with Python bindings via PyO3.

## Repository Structure

```
├── src/                    # Rust source code
├── python/                 # Python package and tests
├── Cargo.toml              # Rust package configuration
├── pyproject.toml          # Python package configuration (maturin)
└── .github/workflows/      # CI/CD workflows
```

## Build Commands

### Rust

```bash
cargo build                 # Build library
cargo test                  # Run Rust tests
cargo doc --open            # Build and view documentation
cargo fmt                   # Format code
cargo clippy                # Lint
```

### Python

```bash
maturin develop             # Build and install locally for development
pytest python/tests/ -v     # Run Python tests
black python/               # Format
flake8 python/              # Lint
mypy python/rustling/       # Type check
```

### Using uv (faster)

```bash
uv run maturin develop
uv run pytest python/tests/ -v
uvx black --check python/
uvx flake8 python/
uvx mypy python/rustling/
```

## Architecture Notes

- **Crate types**: Both `cdylib` (for Python extension) and `rlib` (for `cargo test`)
- **PyO3 features**: `extension-module` is a Cargo feature, enabled by maturin but not during `cargo test`
- **Python bindings**: Rust structs use `#[pyclass]` and methods use `#[pymethods]`
- **Module registration**: `wordseg::register_module()` adds the submodule to Python

## Benchmarking

Benchmark scripts live in `benchmarks/` to compare rustling vs pure Python wordseg:

```bash
# Quick benchmark
python benchmarks/run_wordseg.py --quick

# Full benchmark suite
python benchmarks/run_wordseg.py

# Scaling benchmark with ASCII chart
python benchmarks/scaling.py --plot

# Realistic Chinese text simulation
python benchmarks/realistic.py
```

Requires: `pip install wordseg` for comparison. Build with `--release` for accurate results.

## Conventions

- **Rust docstrings**: Use `///` for public items, `//!` for module-level docs
- **Python docstrings**: NumPy style
- **Type stubs**: `.pyi` files mirror the Python package structure
- **Tests**: Rust tests are inline (`#[cfg(test)]`), Python tests in `python/tests/`

## CI/CD

- **python.yml**: Runs on push/PR. Lint job uses `uvx`, test job runs pytest across Python 3.10-3.14
- **rust.yml**: Runs on push/PR. Format check, clippy, and `cargo test` (requires Python for linking)
- **release.yml**: Triggered on GitHub release. Builds wheels for Linux/macOS/Windows and publishes to PyPI and crates.io

## Release Process

1. Update version in `Cargo.toml` and `pyproject.toml`
2. Create a GitHub release
3. CI automatically publishes to PyPI (trusted publishing) and crates.io (also trusted publishing)
