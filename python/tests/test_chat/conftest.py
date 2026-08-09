import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

CACHE_ROOT = Path.home() / ".rustling"
REPO_ROOT = Path(__file__).resolve().parents[3]
CHATTER_URL = "https://github.com/TalkBank/chatter.git"

# Only these two paths are checked out; the rest of the chatter repo (~42M) is
# never fetched.
SPARSE_PATHS = ("corpus/reference", "spec/errors")


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
    """One invalid-CHAT example extracted from a ``spec/errors`` markdown file."""

    spec_name: str
    code: str
    chat: str
    expected_codes: tuple[str, ...]
    status: str
    # Position within the spec file. A file may hold several examples and they
    # do not all behave alike, so baselines are keyed per example rather than
    # per file.
    example_index: int

    @property
    def key(self) -> str:
        return f"{self.spec_name}#{self.example_index}"


# ``## Example N`` sections hold the CHAT sample; the metadata line above it
# names the codes the sample is expected to trigger.
_CHAT_BLOCK_RE = re.compile(r"```chat\n(.*?)```", re.DOTALL)
_EXPECTED_RE = re.compile(r"\*\*Expected Error Codes\*\*:\s*(.+)")
_STATUS_RE = re.compile(r"\*\*Status\*\*:\s*(\S+)")
# No word boundaries: spec files are named `E241_illegal_untranscribed.md`, and
# `_` is a word character, so `\b` would never match after the digits.
_CODE_RE = re.compile(r"E\d{3}")
_FILENAME_CODE_RE = re.compile(r"^(E\d{3})")


def _parse_error_spec(path: Path) -> list[ErrorSpec]:
    text = path.read_text()
    status_match = _STATUS_RE.search(text)
    status = status_match.group(1) if status_match else "unspecified"
    code_match = _FILENAME_CODE_RE.match(path.name)
    if code_match is None:
        return []
    code = code_match.group(1)

    specs = []
    # Split on example headings so each CHAT block keeps the expected-code line
    # that belongs to it rather than the file's first one.
    sections = re.split(r"\n## +Example\b", text)[1:] or [text]
    for section in sections:
        chat_match = _CHAT_BLOCK_RE.search(section)
        if chat_match is None:
            continue
        expected_match = _EXPECTED_RE.search(section)
        expected = (
            tuple(_CODE_RE.findall(expected_match.group(1)))
            if expected_match
            else (code,)
        )
        specs.append(
            ErrorSpec(
                spec_name=path.name,
                code=code,
                chat=chat_match.group(1),
                expected_codes=expected or (code,),
                status=status,
                example_index=len(specs),
            )
        )
    return specs


@pytest.fixture(scope="session")
def error_specs(chatter_repo_dir):
    """Invalid-CHAT examples from ``spec/errors``, with their expected codes."""
    spec_dir = chatter_repo_dir / "spec" / "errors"
    specs = []
    for path in sorted(spec_dir.glob("*.md")):
        specs.extend(_parse_error_spec(path))
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
