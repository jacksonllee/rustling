"""CoNLL-U (Universal Dependencies) data handling."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from rustling.chat import CHAT

class Token:
    """A single token from a CoNLL-U file (10 tab-separated fields)."""

    @property
    def id(self) -> str:
        """Word index (integer, range like ``"1-2"``, or decimal like ``"1.1"``)."""
        ...

    @property
    def form(self) -> str:
        """Word form or punctuation symbol."""
        ...

    @property
    def lemma(self) -> str:
        """Lemma or stem of the word."""
        ...

    @property
    def upos(self) -> str:
        """Universal POS tag."""
        ...

    @property
    def xpos(self) -> str:
        """Language-specific POS tag, or ``"_"``."""
        ...

    @property
    def feats(self) -> str:
        """Morphological features, or ``"_"``."""
        ...

    @property
    def head(self) -> str:
        """Head of the current word (ID or ``"0"`` for root), or ``"_"``."""
        ...

    @property
    def deprel(self) -> str:
        """Universal dependency relation to HEAD, or ``"_"``."""
        ...

    @property
    def deps(self) -> str:
        """Enhanced dependency graph, or ``"_"``."""
        ...

    @property
    def misc(self) -> str:
        """Any other annotation, or ``"_"``."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class Sentence:
    """A single sentence from a CoNLL-U file."""

    @property
    def comments(self) -> list[str] | None:
        """Comment lines (without the leading ``# ``), or ``None``."""
        ...

    def tokens(self) -> list[Token]:
        """Tokens in this sentence."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class CoNLLU:
    """CoNLL-U (Universal Dependencies) data reader."""

    def __init__(self) -> None: ...
    @classmethod
    def from_strs(
        cls,
        strs: Sequence[str],
        ids: Sequence[str] | None = None,
        parallel: bool = True,
    ) -> CoNLLU:
        """Parse CoNLL-U data from in-memory strings."""
        ...

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | os.PathLike[str]],
        *,
        parallel: bool = True,
    ) -> CoNLLU:
        """Load CoNLL-U data from file paths."""
        ...

    @classmethod
    def from_dir(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".conllu",
        parallel: bool = True,
    ) -> CoNLLU:
        """Recursively load CoNLL-U data from a directory."""
        ...

    @classmethod
    def from_zip(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".conllu",
        parallel: bool = True,
    ) -> CoNLLU:
        """Load CoNLL-U data from a ZIP archive."""
        ...

    @classmethod
    def from_git(
        cls,
        url: str,
        *,
        rev: str | None = None,
        depth: int | None = None,
        match: str | None = None,
        extension: str = ".conllu",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
    ) -> CoNLLU:
        """Load CoNLL-U data from a git repository.

        Clones the repository (or uses a cached clone) and parses all
        matching files from the resulting directory.

        Args:
            url: Git repository URL.
            rev: Branch, tag, or commit hash. If None, uses the
                repository's default branch.
            depth: Clone depth. Defaults to 1 (shallow clone).
                Ignored when rev is a commit hash.
            match: Regex pattern to include only matching file paths.
            extension: File extension to filter by (default: ".conllu").
            cache_dir: Directory for caching cloned repositories.
                Defaults to ``~/.rustling/cache/``.
            force_download: If True, re-clone even if a cached copy exists.
            parallel: If True, use parallel processing.

        Returns:
            A new CoNLL-U reader with the parsed data.
        """
        ...

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        match: str | None = None,
        extension: str = ".conllu",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
    ) -> CoNLLU:
        """Load CoNLL-U data from a URL.

        Downloads the file (or uses a cached copy) and parses it.
        ZIP files are automatically detected and extracted.

        Args:
            url: URL to download from.
            match: Regex pattern to include only matching file paths
                (applicable for ZIP files).
            extension: File extension to filter by (default: ".conllu",
                applicable for ZIP files).
            cache_dir: Directory for caching downloads.
                Defaults to ``~/.rustling/cache/``.
            force_download: If True, re-download even if a cached
                copy exists.
            parallel: If True, use parallel processing.

        Returns:
            A new CoNLL-U reader with the parsed data.
        """
        ...

    @property
    def file_paths(self) -> list[str]:
        """Return the list of file paths."""
        ...

    @property
    def n_files(self) -> int:
        """Return the number of files."""
        ...

    def sentences(self) -> list[Sentence]:
        """Return all sentences across all files as a flat list."""
        ...

    def to_strs(self) -> list[str]:
        """Return CoNLL-U strings, one per file."""
        ...

    def to_chat_strs(self) -> list[str]:
        """Return CHAT format strings, one per file."""
        ...

    def to_chat(self) -> CHAT:
        """Convert to a CHAT object.

        Each CoNLL-U file produces one CHAT file with a default participant
        code ``"SPK"`` (Speaker).

        Returns:
            A :class:`~rustling.chat.CHAT` object.
        """
        ...

    def to_chat_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write CHAT (.cha) files to a directory.

        Args:
            dir_path: Directory path to write .cha files to.
            filenames: Custom filenames for the output files.

        Raises:
            ValueError: If filenames count doesn't match file count.
            IOError: If writing fails.
        """
        ...

    def to_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write CoNLL-U files to a directory.

        Args:
            dir_path: Directory path to write .conllu files to.
            filenames: Custom filenames for the output files.

        Raises:
            ValueError: If filenames count doesn't match file count.
            IOError: If writing fails.
        """
        ...

    def append(self, other: CoNLLU, /) -> None:
        """Append data from another CoNLL-U reader."""
        ...

    def append_left(self, other: CoNLLU, /) -> None:
        """Left-append data from another CoNLL-U reader, preserving order."""
        ...

    def extend(self, others: Sequence[CoNLLU], /) -> None:
        """Extend data from multiple CoNLL-U readers."""
        ...

    def pop(self) -> CoNLLU:
        """Remove and return the last file as a new CoNLL-U reader."""
        ...

    def pop_left(self) -> CoNLLU:
        """Remove and return the first file as a new CoNLL-U reader."""
        ...

    def clear(self) -> None:
        """Remove all data from this reader."""
        ...

    def __add__(self, other: CoNLLU, /) -> CoNLLU: ...
    def __iadd__(self, other: CoNLLU, /) -> CoNLLU: ...
    def __iter__(self) -> Iterator[CoNLLU]: ...
    def __getitem__(self, index: int | slice, /) -> CoNLLU: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

def read_conllu(
    path: str | os.PathLike[str],
    *,
    cls: type[CoNLLU] = CoNLLU,
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
