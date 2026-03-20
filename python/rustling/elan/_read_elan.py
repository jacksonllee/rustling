from __future__ import annotations

import os

from typing import TYPE_CHECKING

from rustling._lib_name import elan as _elan

if TYPE_CHECKING:
    from rustling.elan import ELAN


def read_elan(
    path: str | os.PathLike[str],
    *,
    cls: type[ELAN] = _elan.ELAN,
) -> ELAN:
    """Read ELAN data.

    Args:
        path: Path to a ``.zip`` file, a local directory containing ``.eaf``
            files, a single ``.eaf`` file, a git repository URL
            (ending in ``.git``), or an HTTP/HTTPS URL.
        cls: The class used to create the reader. Must be ``ELAN`` or a
            subclass of it.

    Returns:
        An ``ELAN`` instance.

    Raises:
        TypeError: If *cls* is not ``ELAN`` or a subclass of it.
        ValueError: If *path* does not point to a recognized source.
    """
    if cls != _elan.ELAN and not issubclass(cls, _elan.ELAN):
        raise TypeError(f"Only an ELAN class or its child class is allowed: {cls}")

    path = os.fspath(path)
    path_lower = path.lower()
    if path_lower.startswith(("http://", "https://")) and path_lower.endswith(".git"):
        return cls.from_git(path)
    elif path_lower.startswith(("http://", "https://")):
        return cls.from_url(path)
    elif path_lower.endswith(".zip"):
        return cls.from_zip(path)
    elif os.path.isdir(path):
        return cls.from_dir(path)
    elif path_lower.endswith(".eaf"):
        return cls.from_files([path])
    else:
        raise ValueError(
            "path is not one of the accepted choices of "
            f"{{.zip file, local directory, .eaf file, git URL, HTTP URL}}: {path}"
        )
