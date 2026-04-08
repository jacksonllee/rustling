"""SRT (SubRip Subtitle) data handling."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from rustling.chat import CHAT
    from rustling.elan import ELAN
    from rustling.textgrid import TextGrid

class Utterance:
    """A single subtitle block within an SRT file."""

    def __init__(
        self,
        *,
        index: int,
        line: str,
        time_marks: tuple[int, int],
    ) -> None:
        """Create an Utterance.

        Args:
            index: 1-based sequence number.
            line: The subtitle text.
            time_marks: Start and end time in milliseconds.
        """
        ...

    @property
    def index(self) -> int:
        """1-based sequence number from the SRT file."""
        ...

    @property
    def line(self) -> str:
        """The subtitle text."""
        ...

    @property
    def time_marks(self) -> tuple[int, int]:
        """Start and end time in milliseconds as a tuple."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class SRT:
    """SRT (SubRip Subtitle) data reader."""

    def __init__(self) -> None: ...
    @classmethod
    def from_strs(
        cls,
        strs: Sequence[str],
        ids: Sequence[str] | None = None,
        parallel: bool = True,
    ) -> SRT:
        """Parse SRT data from in-memory strings."""
        ...

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | os.PathLike[str]],
        *,
        parallel: bool = True,
    ) -> SRT:
        """Load SRT data from file paths."""
        ...

    @classmethod
    def from_dir(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".srt",
        parallel: bool = True,
    ) -> SRT:
        """Recursively load SRT data from a directory."""
        ...

    @classmethod
    def from_zip(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".srt",
        parallel: bool = True,
    ) -> SRT:
        """Load SRT data from a ZIP archive."""
        ...

    @classmethod
    def from_git(
        cls,
        url: str,
        *,
        rev: str | None = None,
        depth: int | None = None,
        match: str | None = None,
        extension: str = ".srt",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
    ) -> SRT:
        """Load SRT data from a git repository.

        Clones the repository (or uses a cached clone) and parses all
        matching files from the resulting directory.

        Args:
            url: Git repository URL.
            rev: Branch, tag, or commit hash. If None, uses the
                repository's default branch.
            depth: Clone depth. Defaults to 1 (shallow clone).
                Ignored when rev is a commit hash.
            match: Regex pattern to include only matching file paths.
            extension: File extension to filter by (default: ".srt").
            cache_dir: Directory for caching cloned repositories.
                Defaults to ``~/.rustling/cache/``.
            force_download: If True, re-clone even if a cached copy exists.
            parallel: If True, use parallel processing.

        Returns:
            A new SRT reader with the parsed data.
        """
        ...

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        match: str | None = None,
        extension: str = ".srt",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
    ) -> SRT:
        """Load SRT data from a URL.

        Downloads the file (or uses a cached copy) and parses it.
        ZIP files are automatically detected and extracted.

        Args:
            url: URL to download from.
            match: Regex pattern to include only matching file paths
                (applicable for ZIP files).
            extension: File extension to filter by (default: ".srt",
                applicable for ZIP files).
            cache_dir: Directory for caching downloads.
                Defaults to ``~/.rustling/cache/``.
            force_download: If True, re-download even if a cached
                copy exists.
            parallel: If True, use parallel processing.

        Returns:
            A new SRT reader with the parsed data.
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

    def utterances(self) -> list[Utterance]:
        """Return all subtitle blocks across all files as a flat list."""
        ...

    def to_strs(self) -> list[str]:
        """Return SRT strings, one per file."""
        ...

    def to_chat_strs(self) -> list[str]:
        """Return CHAT format strings, one per file."""
        ...

    def to_chat(self) -> CHAT:
        """Convert to a CHAT object.

        Each SRT file produces one CHAT file with a default participant
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

    def to_elan_strs(self) -> list[str]:
        """Return EAF XML strings, one per file."""
        ...

    def to_elan(self) -> ELAN:
        """Convert to an ELAN object.

        Each SRT file produces one ELAN file with a single tier
        named ``"SPK"`` (Speaker).

        Returns:
            A :class:`~rustling.elan.ELAN` object.
        """
        ...

    def to_elan_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write ELAN (.eaf) files to a directory.

        Args:
            dir_path: Directory path to write .eaf files to.
            filenames: Custom filenames for the output files.

        Raises:
            ValueError: If filenames count doesn't match file count.
            IOError: If writing fails.
        """
        ...

    def to_textgrid_strs(self) -> list[str]:
        """Return TextGrid format strings, one per file."""
        ...

    def to_textgrid(self) -> TextGrid:
        """Convert to a TextGrid object."""
        ...

    def to_textgrid_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write TextGrid (.TextGrid) files to a directory."""
        ...

    def to_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write SRT files to a directory.

        Args:
            dir_path: Directory path to write .srt files to.
            filenames: Custom filenames for the output files.

        Raises:
            ValueError: If filenames count doesn't match file count.
            IOError: If writing fails.
        """
        ...

    def append(self, other: SRT, /) -> None:
        """Append data from another SRT reader."""
        ...

    def append_left(self, other: SRT, /) -> None:
        """Left-append data from another SRT reader, preserving order."""
        ...

    def extend(self, others: Sequence[SRT], /) -> None:
        """Extend data from multiple SRT readers."""
        ...

    def pop(self) -> SRT:
        """Remove and return the last file as a new SRT reader."""
        ...

    def pop_left(self) -> SRT:
        """Remove and return the first file as a new SRT reader."""
        ...

    def clear(self) -> None:
        """Remove all data from this reader."""
        ...

    def __add__(self, other: SRT, /) -> SRT: ...
    def __iadd__(self, other: SRT, /) -> SRT: ...
    def __iter__(self) -> Iterator[SRT]: ...
    def __getitem__(self, index: int | slice, /) -> SRT: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

def read_srt(
    path: str | os.PathLike[str],
    *,
    cls: type[SRT] = SRT,
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
