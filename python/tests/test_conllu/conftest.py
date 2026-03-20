import subprocess
from pathlib import Path

import pytest

UD_ENGLISH_EWT_DIR = Path.home() / ".rustling" / "ud-english-ewt"
UD_CANTONESE_HK_DIR = Path.home() / ".rustling" / "ud-cantonese-hk"


@pytest.fixture(scope="session")
def ud_english_ewt_dir():
    """Download UD_English-EWT data if not present."""
    if not UD_ENGLISH_EWT_DIR.exists():
        UD_ENGLISH_EWT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/UniversalDependencies/UD_English-EWT.git",
                str(UD_ENGLISH_EWT_DIR),
            ],
            check=True,
        )
    return UD_ENGLISH_EWT_DIR


@pytest.fixture(scope="session")
def ud_cantonese_hk_dir():
    """Download UD_Cantonese-HK data if not present."""
    if not UD_CANTONESE_HK_DIR.exists():
        UD_CANTONESE_HK_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/UniversalDependencies/UD_Cantonese-HK.git",
                str(UD_CANTONESE_HK_DIR),
            ],
            check=True,
        )
    return UD_CANTONESE_HK_DIR
