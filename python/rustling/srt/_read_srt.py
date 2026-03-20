from __future__ import annotations

import os

from typing import TYPE_CHECKING

from rustling._lib_name import srt as _srt

if TYPE_CHECKING:
    from rustling.srt import SRT


def read_srt(
    path: str | os.PathLike[str],
    *,
    cls: type[SRT] = _srt.SRT,
) -> SRT:
    """Read SRT data.

    Args:
        path: Path to a ``.zip`` file, a local directory containing ``.srt``
            files, a single ``.srt`` file, a git repository URL
            (ending in ``.git``), or an HTTP/HTTPS URL.
        cls: The class used to create the reader. Must be ``SRT`` or a
            subclass of it.

    Returns:
        An ``SRT`` instance.

    Raises:
        TypeError: If *cls* is not ``SRT`` or a subclass of it.
        ValueError: If *path* does not point to a recognized source.
    """
    if cls != _srt.SRT and not issubclass(cls, _srt.SRT):
        raise TypeError(f"Only an SRT class or its child class is allowed: {cls}")

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
    elif path_lower.endswith(".srt"):
        return cls.from_files([path])
    else:
        raise ValueError(
            "path is not one of the accepted choices of "
            f"{{.zip file, local directory, .srt file, git URL, HTTP URL}}: {path}"
        )
