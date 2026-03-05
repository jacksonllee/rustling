# AGENTS.md

This file provides guidance for developing this repository.

## Project Overview

**Rustling** is a computational linguistics library implemented in Rust with Python bindings via PyO3.

## Repository Structure

```
├── src/                    # Rust source code
├── python/                 # Python package and tests
├── Cargo.toml              # Rust package configuration
├── pyproject.toml          # Python package configuration (maturin)
└── .github/workflows/      # CI/CD workflows
```

## Local Dev Setup

### Prerequisites

- **Rust** (stable toolchain)
- **Python 3.10+** with [`uv`](https://docs.astral.sh/uv/)
- **flatc** (FlatBuffers compiler) — recommended for developers.
  `build.rs` invokes `flatc --rust` to generate code from `.fbs` schemas.
  If `flatc` is not found, `build.rs` falls back to pre-committed `model_generated.rs`
  files, so downstream Rust consumers can build without installing `flatc`.
  Developers who modify `.fbs` schemas **must** have `flatc` installed to regenerate
  the code and re-commit the updated `model_generated.rs` files.
  The project pins **v25.12.19**. On macOS:
  ```bash
  brew install flatbuffers
  ```
  Note: Homebrew may install a newer version; the pinned version is v25.12.19
  ([download](https://github.com/google/flatbuffers/releases/tag/v25.12.19)).

### Pre-commit hooks

The repo includes a `.pre-commit-config.yaml` with the following hooks:

| Hook | What it checks |
|------|---------------|
| `cargo fmt` | Rust formatting |
| `cargo clippy` | Rust lints (warnings = errors) |
| `black` | Python formatting |
| `flake8` | Python style |

To enable:

```bash
pre-commit install
```

Hooks run automatically on `git commit`. To run manually against all files:

```bash
pre-commit run --all-files
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

`uv` manages the virtual environment.

```bash
uv run maturin develop                     # Build and install locally for development
uv run pytest python/tests/ -v                          # Run Python tests
uvx black python/                                       # Format
uvx flake8 python/                                      # Lint
uvx mypy python/rustling/                               # Type check
uv run python -m mypy.stubtest rustling --concise --ignore-missing-stub --allowlist python/stubtest_allowlist.txt  # Verify stubs match runtime
cd python/docs && make clean && make html && cd ../...  # Build documentation
```

## Architecture

- **Separate layers of Rust and Python**:
  Rustling is fully usable as a Rust library, without crossing the Python/PyO3 boundary.
  Anything that needs to be exposed to the Python bindings is built as a separate layer
  on top of the Rust code. See [Rust/Python Layering Pattern](#rustpython-layering-pattern) below.
- **Models**: Implemented models as Rust structs / Python classes use these method names
  (depending on the model, not all methods are available):
   * `fit` for training a model.
     `fit(data, labels=None)` for a model that supports semi-supervised learning,
     `fit(data, labels)` for a model for supervised leanring only, and
     `fit(data)` for a model for unsupervised learning only.
   * `predict` for making predictions for unlabeled data
   * `score` for scoring an observation and its labels using the given model
   * `save` for saving the trained model on disk as a FlatBuffers binary (`.bin`) file
   * `load` for loading from disk (either str or os.PathLike) a trained model saved by the `save` method
- **Model persistence**: All models use **FlatBuffers binary** (`.bin` files)
  compressed with **zstd** (level 19). No JSON, no pickle, no gzip.
  Model floats are stored as **f32** on disk (`[float]` in FlatBuffers schema);
  internal computation stays f64. The f32→f64 widening cast on load is
  negligible overhead.
- **FlatBuffers schemas**: Each module with a model has a `model.fbs` schema
  co-located with its `mod.rs` (e.g., `src/hmm/model.fbs`). Generated Rust code
  goes to Cargo's `OUT_DIR` via `build.rs` — generated files are NOT committed.
  `build.rs` invokes `flatc --rust` for each schema. `flatc` is pinned to v25.12.19.
- **Shared persistence helpers** in `src/persistence.rs`:
  `read_all_bytes` and `flatbuffers_verifier_opts()`.
- **Crate types**: `Cargo.toml` declares only `rlib`; maturin adds `cdylib` automatically when building the Python extension
- **PyO3 features**: `extension-module` is a Cargo feature, enabled by maturin but not during `cargo test`
- **Python bindings**: Rust structs use `#[pyclass]` and methods use `#[pymethods]`
- **Module registration**: `wordseg::register_module()` adds the submodule to Python

## Rust/Python Layering Pattern

Every major feature follows a three-layer pattern: **trait** (logic) → **Rust struct** (pure Rust API) → **PyO3 wrapper** (Python API). This keeps business logic in one place and makes the library fully usable from both Rust and Python. Use the hypothetical `Foo` below as a model when adding new features.

### Layer 1: Trait with default methods

Define a trait. Required methods describe the minimal
storage contract; default methods provide all the shared logic "for free."

```rust
// Hypothetically, in src/foo.rs

pub trait BaseFoo: Sized {
    // --- Required: each concrete type must implement these ---
    fn items(&self) -> &Vec<Item>;
    fn items_mut(&mut self) -> &mut Vec<Item>;
    fn from_items(items: Vec<Item>) -> Self;

    // --- Default: shared logic, available to both Foo and PyFoo ---
    fn len(&self) -> usize {
        self.items().len()
    }
    fn process(&self, kind: &str) -> Vec<f64> {
        // Business logic lives here, not in the pymethod wrappers.
        self.items().iter().map(|item| item.compute(kind)).collect()
    }
    fn filter(&self, predicate: &str) -> Result<Self, String> {
        let filtered = self.items().iter()
            .filter(|i| i.matches(predicate))
            .cloned()
            .collect();
        Ok(Self::from_items(filtered))
    }
}
```

### Layer 2: Pure Rust struct

The Rust struct is what Rust-only users (and downstream crates) consume directly.
It implements the trait and adds Rust-specific constructors and I/O.

```rust
// In src/foo.rs (same file)

/// Pure Rust struct. For Python, use PyFoo.
#[derive(Clone, Debug)]
pub struct Foo {
    pub(crate) items: Vec<Item>,
}

impl BaseFoo for Foo {
    fn items(&self) -> &Vec<Item> { &self.items }
    fn items_mut(&mut self) -> &mut Vec<Item> { &mut self.items }
    fn from_items(items: Vec<Item>) -> Self { Self { items } }
}

impl Foo {
    /// Rust-specific factory methods, parsing, I/O, etc.
    pub fn from_strs(strs: Vec<String>) -> Self {
        let items = strs.into_iter().map(|s| Item::parse(&s)).collect();
        Self { items }
    }
    pub fn read_file(path: &str) -> Result<Self, std::io::Error> {
        // ...
    }
}
```

### Layer 3: PyO3 wrapper (composition)

The `#[pyclass]` wrapper holds the Rust struct as an `inner` field.
It implements the same trait by delegating to `inner`, and its `#[pymethods]`
are thin wrappers that call the trait's default methods.

```rust
// In src/foo.rs (same file)

/// Python-exposed wrapper. Python users see this as the class `Foo`.
#[pyclass(name = "Foo")]
#[derive(Clone)]
pub struct PyFoo {
    pub inner: Foo,  // Composition, not inheritance
}

// Implement the trait by delegating to inner.
impl BaseFoo for PyFoo {
    fn items(&self) -> &Vec<Item> { self.inner.items() }
    fn items_mut(&mut self) -> &mut Vec<Item> { self.inner.items_mut() }
    fn from_items(items: Vec<Item>) -> Self {
        Self { inner: Foo::from_items(items) }
    }
}

#[pymethods]
impl PyFoo {
    #[new]
    fn new() -> Self {
        Self::from_items(Vec::new())
    }

    // Classmethods delegate to Foo's constructors, then wrap.
    #[classmethod]
    #[pyo3(signature = (strs))]
    fn from_strs(_cls: &Bound<'_, PyType>, strs: Vec<String>) -> PyResult<Self> {
        let foo = Foo::from_strs(strs);
        Ok(Self { inner: foo })
    }

    // Thin wrappers: delegate straight to trait default methods.
    #[getter]
    fn len(&self) -> usize {
        BaseFoo::len(self)
    }

    #[pyo3(signature = (*, kind="default"))]
    fn process(&self, kind: &str) -> Vec<f64> {
        BaseFoo::process(self, kind)
    }

    #[pyo3(signature = (*, predicate=None))]
    fn filter(&self, predicate: Option<&str>) -> PyResult<Self> {
        match predicate {
            Some(p) => BaseFoo::filter(self, p)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e)),
            None => Ok(self.clone()),
        }
    }
}
```

### Python side

```python
# python/rustling/foo/__init__.py
from rustling._lib_name import foo as _foo

Foo = _foo.Foo
```

```python
# python/rustling/foo/__init__.pyi
class Foo:
    def __init__(self) -> None: ...

    @classmethod
    def from_strs(cls, strs: Sequence[str]) -> Foo: ...

    @property
    def len(self) -> int: ...
    def process(self, *, kind: str = "default") -> list[float]: ...
    def filter(self, *, predicate: str | None = None) -> Foo: ...
```

### Module registration

```rust
// src/foo/mod.rs
pub(crate) fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let foo_module = PyModule::new(parent_module.py(), "foo")?;
    foo_module.add_class::<PyFoo>()?;
    parent_module.add_submodule(&foo_module)?;
    Ok(())
}
```

Called from `lib.rs`:

```rust
foo::register_module(m)?;
```

### Why this pattern?

| Concern | Where it lives |
|---|---|
| Business logic | Trait default methods (`BaseFoo`) |
| Storage contract | Trait required methods |
| Rust-only API | `impl Foo` (constructors, I/O) |
| Python API | `#[pymethods] impl PyFoo` (thin wrappers) |
| Error conversion, Python types | PyFoo's pymethods only |

- **One source of truth for logic**: changing a trait default method updates both Rust and Python.
- **Pure Rust is first-class**: `Foo` works without Python; downstream Rust crates depend on it.
- **Thin Python layer**: PyFoo methods are ~1-line delegations, minimizing binding bugs.
- **Performance exception**: For performance-critical methods (e.g., `PyHiddenMarkovModel::predict`), the `#[pymethods]` block may override the trait default with a specialized implementation that avoids PyO3 conversion overhead (zero-copy string access, pre-encoded flat arrays, etc.).
- **Testable at both levels**: `cargo test` covers `Foo`; `pytest` covers `PyFoo`.

## Benchmarking

Benchmark scripts live in `benchmarks/` to compare Rustling vs Python implementations.
See `benchmarks/README.md`.

## Conventions

- **Rust docstrings**: Use `///` for public items, `//!` for module-level docs
- **Rust code style**: Must pass `cargo fmt` and `cargo clippy` (enforced by GitHub Actions CI)
- **Python code style**: Must pass `black` and `flake8` for `python/rustling/` and `benchmarks/` (enforced by GitHub Actions CI)
- **Python docstrings**: Use Google style
- **Type stubs**: `.pyi` files mirror the Python package structure
- **Tests**: Rust tests are inline (`#[cfg(test)]`), Python tests in `python/tests/`

## CI/CD

CI/CD is on GitHub Actions, see `.github/workflows/`.
All jobs that build Rust code install **flatc v25.12.19** (FlatBuffers compiler)
before the build step, since `build.rs` invokes `flatc` to generate Rust code
from `.fbs` schemas.

- **python.yml**: Runs on push/PR. Lint job uses `uvx`, test job runs pytest across Python 3.10-3.14
- **rust.yml**: Runs on push/PR. Format check, clippy, and `cargo test` (requires Python for linking)
- **release.yml**: Triggered on GitHub release. Builds wheels for Linux/macOS/Windows and publishes to PyPI and crates.io

## Release Process

1. Update version in `Cargo.toml` and `pyproject.toml`.
2. Create a GitHub release.
3. CI automatically publishes to PyPI (trusted publishing) and crates.io (also trusted publishing).
