import os
import subprocess
from pathlib import Path

import pytest

CANTOMAP_DIR = Path.home() / ".rustling" / "cantomap"


@pytest.fixture(scope="session")
def cantomap_dir():
    """Download CantoMap data if not present, skipping LFS files."""
    if not CANTOMAP_DIR.exists():
        CANTOMAP_DIR.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # The .wav audio files are versioned with Git LFS,
        # but we don't need them (and they can't be checked out
        # properly anyway, since the repo's LFS credits
        # have run out apparently).
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/gwinterstein/CantoMap.git",
                str(CANTOMAP_DIR),
            ],
            check=True,
            env=env,
        )
    return CANTOMAP_DIR
