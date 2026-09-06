import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # chatter's error specs are TOML; tomllib landed in 3.11.
    import tomli as tomllib

CACHE_ROOT = Path.home() / ".rustling"
REPO_ROOT = Path(__file__).resolve().parents[3]
CHATTER_URL = "https://github.com/TalkBank/chatter.git"

# Only these paths are checked out; the rest of the chatter repo (~42M) is
# never fetched. `spec/codes` holds the error-code registry, which is where a
# code's status lives; `spec/errors` holds the examples that document it.
SPARSE_PATHS = ("corpus/reference", "spec/errors", "spec/codes")


def _pinned_chatter_tag() -> str:
    """Read the ``chatter`` git tag that Cargo.toml pins.

    Deriving the tag rather than hard-coding it keeps the test corpus and the
    parser under test on the same version: bumping the dependency moves the
    fixtures with it, and a stale checkout can never be silently reused.
    """
    cargo_toml = (REPO_ROOT / "Cargo.toml").read_text()
    match = re.search(r'talkbank-model\s*=\s*\{[^}]*\btag\s*=\s*"([^"]+)"', cargo_toml)
    if match is None:
        raise RuntimeError(
            f"no pinned talkbank-model tag found in {REPO_ROOT / 'Cargo.toml'}"
        )
    return match.group(1)


@pytest.fixture(scope="session")
def chatter_repo_dir():
    """Sparse checkout of the pinned ``chatter`` tag, cached under ~/.rustling.

    ``corpus/reference`` and ``spec/errors`` are the two directories rustling
    tests against. They are declared license-clear for redistribution, but are
    fetched rather than vendored so they cannot drift from the pinned parser.
    """
    tag = _pinned_chatter_tag()
    dest = CACHE_ROOT / f"chatter-{tag}"
    marker = dest / "corpus" / "reference"

    if not marker.exists():
        # A previous run may have failed partway through; start clean.
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    tag,
                    "--filter=blob:none",
                    "--sparse",
                    CHATTER_URL,
                    str(dest),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dest), "sparse-checkout", "set", *SPARSE_PATHS],
                check=True,
            )
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise

    return dest


@pytest.fixture(scope="session")
def reference_corpus_dir(chatter_repo_dir):
    """chatter's reference corpus: constructed CHAT files that must all parse.

    This is chatter's own 100%-pass gate, so every file here is valid CHAT and
    rustling is expected to load all of them under ``strict=True``.
    """
    return chatter_repo_dir / "corpus" / "reference"


@pytest.fixture(scope="session")
def reference_corpus_files(reference_corpus_dir):
    """Every reference ``.cha`` file, sorted. The corpus is nested by topic."""
    return sorted(reference_corpus_dir.rglob("*.cha"))


@dataclass(frozen=True)
class ErrorSpec:
    """One CHAT example extracted from a ``spec/errors`` file.

    Since chatter's spec format moved to TOML frontmatter, each example states
    a ``claim`` about itself rather than listing the codes it expects, and the
    claim's negative half is part of the contract: a ``subsumed_by`` example
    asserts that a more general rule fires and that the spec's own code does
    *not*.
    """

    spec_name: str
    code: str
    chat: str
    status: str
    level: str
    # ``violates`` (the spec's own code must fire), ``legal`` (the spec's own
    # code must not fire -- which does not promise the example is clean, only
    # that it satisfies this one rule), or ``subsumed`` (some other code fires
    # and this spec's own code must not).
    claim: str
    # The codes a ``subsumed`` example expects instead of its own; empty for
    # the other two claims.
    subsumed_by: tuple[str, ...]
    # Position within the spec file. A file may hold several examples and they
    # do not all behave alike, so baselines are keyed per example rather than
    # per file.
    example_index: int

    @property
    def key(self) -> str:
        return f"{self.spec_name}#{self.example_index}"

    @property
    def expected_codes(self) -> tuple[str, ...]:
        """The codes loading this example must report, if any."""
        if self.claim == "violates":
            return (self.code,)
        return self.subsumed_by


# Frontmatter runs from the opening `+++` to the next line that is exactly
# `+++`; everything the tests need lives inside it, and the markdown body after
# it is prose. Every spec file has exactly one such block, so a file without
# one (the enhancement guide) is not a spec.
_FRONTMATTER_RE = re.compile(r"\A\+\+\+\n(.*?)\n\+\+\+$", re.DOTALL | re.MULTILINE)
# No word boundaries: spec files are named `E241_illegal_untranscribed.md`, and
# `_` is a word character, so `\b` would never match after the digits.
_CODE_RE = re.compile(r"E\d{3}")


def _code_statuses(chatter_repo_dir: Path) -> dict[str, str]:
    """Every error code's status, from chatter's code registry.

    ``spec/codes/error-codes.toml`` is the declared single owner of a code's
    status; the spec files under ``spec/errors`` only document a code and no
    longer carry it. Reading it here rather than from a spec's frontmatter is
    what keeps the two from drifting.
    """
    registry = chatter_repo_dir / "spec" / "codes" / "error-codes.toml"
    data = tomllib.loads(registry.read_text())
    return {entry["code"]: entry["status"] for entry in data.get("code", [])}


def _parse_error_spec(path: Path, statuses: dict[str, str]) -> list[ErrorSpec]:
    match = _FRONTMATTER_RE.match(path.read_text())
    if match is None:
        return []
    data = tomllib.loads(match.group(1))
    code = data.get("code", "")
    # Skips the `E###` template in README.md, and `W1xx` warning specs: a
    # warning is not an error, and `strict=True` raises only on errors, so a
    # warning example could never do anything but sit in a baseline.
    if not _CODE_RE.fullmatch(code):
        return []
    # A spec's code is a foreign key into the registry, so a spec naming a code
    # the registry does not define is a broken checkout rather than a code to
    # quietly treat as enforced.
    if code not in statuses:
        raise RuntimeError(
            f"{path.name} documents {code}, which is absent from chatter's "
            f"error-code registry"
        )
    status = statuses[code]

    specs = []
    for example in data.get("example", []):
        chat = example.get("chat")
        if chat is None:
            continue
        claim = example.get("claim", "violates")
        if isinstance(claim, dict):
            subsumed_by = claim.get("subsumed_by", [])
            if isinstance(subsumed_by, str):
                subsumed_by = [subsumed_by]
            claim_kind = "subsumed"
        else:
            subsumed_by = []
            claim_kind = claim
        specs.append(
            ErrorSpec(
                spec_name=path.name,
                code=code,
                chat=chat,
                status=status,
                level=example.get("level", "unspecified"),
                claim=claim_kind,
                subsumed_by=tuple(subsumed_by),
                example_index=len(specs),
            )
        )
    return specs


@pytest.fixture(scope="session")
def error_specs(chatter_repo_dir):
    """Every CHAT example in ``spec/errors``, with the claim it makes."""
    spec_dir = chatter_repo_dir / "spec" / "errors"
    statuses = _code_statuses(chatter_repo_dir)
    specs = []
    for path in sorted(spec_dir.glob("*.md")):
        specs.extend(_parse_error_spec(path, statuses))
    return specs


PRIVATE_DATA_DIR = CACHE_ROOT / "private-test-data"


@pytest.fixture(scope="session")
def private_data_dir():
    """Provide private test data, downloading via gh if needed.

    On first local run, requires the environment variable:
      - PRIVATE_TEST_REPO: GitHub repo in "owner/repo" format
    """
    if PRIVATE_DATA_DIR.exists():
        # Try to pull latest changes; ignore failures (e.g., offline)
        subprocess.run(
            ["git", "-C", str(PRIVATE_DATA_DIR), "pull"],
            capture_output=True,
        )
        return PRIVATE_DATA_DIR

    repo = os.environ.get("PRIVATE_TEST_REPO")

    if not repo:
        pytest.skip("PRIVATE_TEST_REPO not set")

    if not shutil.which("gh"):
        pytest.skip("gh CLI not available")

    result = subprocess.run(
        [
            "gh",
            "repo",
            "clone",
            repo,
            str(PRIVATE_DATA_DIR),
            "--",
            "--depth",
            "1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to clone private repo: {result.stderr.decode().strip()}")

    return PRIVATE_DATA_DIR
