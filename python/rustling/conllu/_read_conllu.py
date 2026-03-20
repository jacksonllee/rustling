from __future__ import annotations

import os

from typing import TYPE_CHECKING

from rustling._lib_name import conllu as _conllu

if TYPE_CHECKING:
    from rustling.conllu import CoNLLU


def read_conllu(
    path: str | os.PathLike[str],
    *,
    cls: type[CoNLLU] = _conllu.CoNLLU,
) -> CoNLLU:
    """Read CoNLL-U data.

    Args:
        path: Path to a ``.zip`` file, a local directory containing ``.conllu``
            files, a single ``.conllu`` file, a git repository URL
            (ending in ``.git``), or an HTTP/HTTPS URL.
        cls: The class used to create the reader. Must be ``CoNLLU`` or a
            subclass of it.

    Returns:
        A ``CoNLLU`` instance.

    Raises:
        TypeError: If *cls* is not ``CoNLLU`` or a subclass of it.
        ValueError: If *path* does not point to a recognized source.
    """
    if cls != _conllu.CoNLLU and not issubclass(cls, _conllu.CoNLLU):
        raise TypeError(f"Only a CoNLLU class or its child class is allowed: {cls}")

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
    elif path_lower.endswith(".conllu"):
        return cls.from_files([path])
    else:
        raise ValueError(
            "path is not one of the accepted choices of "
            f"{{.zip file, local directory, .conllu file, git URL, HTTP URL}}: {path}"
        )
