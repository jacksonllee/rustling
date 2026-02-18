import os
import shutil
import subprocess
from pathlib import Path

import pytest

TESTCHAT_DIR = Path.home() / ".rustling" / "testchat"


@pytest.fixture(scope="session")
def testchat_good_dir():
    """Download TalkBank testchat data if not present."""
    good_dir = TESTCHAT_DIR / "good"
    if not good_dir.exists():
        TESTCHAT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/TalkBank/testchat.git",
                str(TESTCHAT_DIR),
            ],
            check=True,
        )
    return good_dir


@pytest.fixture(scope="session")
def testchat_bad_dir(testchat_good_dir):
    # The testchat_good_dir parameter is intentional — it ensures
    # the testchat repo is cloned before we try to access the bad/ directory.
    """Provide testchat/bad directory (cloned alongside good)."""
    return TESTCHAT_DIR / "bad"


PRIVATE_DATA_DIR = Path.home() / ".rustling" / "private-test-data"


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
