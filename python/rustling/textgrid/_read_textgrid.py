from __future__ import annotations

import os

from typing import TYPE_CHECKING

from rustling._lib_name import textgrid as _textgrid

if TYPE_CHECKING:
    from rustling.textgrid import TextGrid


def read_textgrid(
    path: str | os.PathLike[str],
    *,
    cls: type[TextGrid] = _textgrid.TextGrid,
) -> TextGrid:
    """Read TextGrid data.

    Args:
        path: Path to a ``.zip`` file, a local directory containing
            ``.TextGrid`` files, a single ``.TextGrid`` file, a git
            repository URL (ending in ``.git``), or an HTTP/HTTPS URL.
        cls: The class used to create the reader. Must be ``TextGrid`` or a
            subclass of it.

    Returns:
        A ``TextGrid`` instance.

    Raises:
        TypeError: If *cls* is not ``TextGrid`` or a subclass of it.
        ValueError: If *path* does not point to a recognized source.
    """
    if cls != _textgrid.TextGrid and not issubclass(cls, _textgrid.TextGrid):
        raise TypeError(f"Only a TextGrid class or its child class is allowed: {cls}")

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
    elif path_lower.endswith(".textgrid"):
        return cls.from_files([path])
    else:
        raise ValueError(
            "path is not one of the accepted choices of "
            f"{{.zip file, local directory, .TextGrid file, git URL, HTTP URL}}: "
            f"{path}"
        )
