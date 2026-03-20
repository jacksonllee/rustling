"""TextGrid (Praat) file parsing."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from rustling.chat import CHAT
    from rustling.elan import ELAN
    from rustling.srt import SRT

class Interval:
    """A single interval within an IntervalTier."""

    @property
    def xmin(self) -> float:
        """Start time in seconds."""
        ...

    @property
    def xmax(self) -> float:
        """End time in seconds."""
        ...

    @property
    def text(self) -> str:
        """The annotation text."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Point:
    """A single point within a TextTier (PointTier)."""

    @property
    def number(self) -> float:
        """Time in seconds."""
        ...

    @property
    def mark(self) -> str:
        """The annotation text."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class IntervalTier:
    """An interval tier within a TextGrid file."""

    @property
    def name(self) -> str:
        """Tier name."""
        ...

    @property
    def xmin(self) -> float:
        """Start time in seconds."""
        ...

    @property
    def xmax(self) -> float:
        """End time in seconds."""
        ...

    @property
    def intervals(self) -> list[Interval]:
        """Intervals in this tier."""
        ...

    @property
    def tier_class(self) -> str:
        """Tier class: always ``"IntervalTier"``."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class TextTier:
    """A text tier (PointTier) within a TextGrid file."""

    @property
    def name(self) -> str:
        """Tier name."""
        ...

    @property
    def xmin(self) -> float:
        """Start time in seconds."""
        ...

    @property
    def xmax(self) -> float:
        """End time in seconds."""
        ...

    @property
    def points(self) -> list[Point]:
        """Points in this tier."""
        ...

    @property
    def tier_class(self) -> str:
        """Tier class: always ``"TextTier"``."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class TextGrid:
    """TextGrid (Praat) data reader."""

    def __init__(self) -> None: ...
    @classmethod
    def from_strs(
        cls,
        strs: Sequence[str],
        ids: Sequence[str] | None = None,
        parallel: bool = True,
    ) -> TextGrid:
        """Parse TextGrid data from in-memory strings."""
        ...

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | os.PathLike[str]],
        *,
        parallel: bool = True,
    ) -> TextGrid:
        """Load TextGrid data from file paths."""
        ...

    @classmethod
    def from_dir(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".TextGrid",
        parallel: bool = True,
    ) -> TextGrid:
        """Recursively load TextGrid data from a directory."""
        ...

    @classmethod
    def from_zip(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".TextGrid",
        parallel: bool = True,
    ) -> TextGrid:
        """Load TextGrid data from a ZIP archive."""
        ...

    @classmethod
    def from_git(
        cls,
        url: str,
        *,
        rev: str | None = None,
        depth: int | None = None,
        match: str | None = None,
        extension: str = ".TextGrid",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
    ) -> TextGrid:
        """Load TextGrid data from a git repository."""
        ...

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        match: str | None = None,
        extension: str = ".TextGrid",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
    ) -> TextGrid:
        """Load TextGrid data from a URL."""
        ...

    @property
    def file_paths(self) -> list[str]:
        """Return the list of file paths."""
        ...

    @property
    def n_files(self) -> int:
        """Return the number of files."""
        ...

    def tiers(self) -> list[list[IntervalTier | TextTier]]:
        """Return tiers as a list of lists, one list per file."""
        ...

    def to_strs(self) -> list[str]:
        """Return TextGrid strings, one per file."""
        ...

    def to_chat_strs(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> list[str]:
        """Return CHAT format strings, one per file."""
        ...

    def to_chat(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> CHAT:
        """Convert to a CHAT object."""
        ...

    def to_chat_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        participants: Sequence[str] | None = None,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write CHAT (.cha) files to a directory."""
        ...

    def to_elan_strs(self) -> list[str]:
        """Return EAF XML strings, one per file."""
        ...

    def to_elan(self) -> ELAN:
        """Convert to an ELAN object."""
        ...

    def to_elan_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write ELAN (.eaf) files to a directory."""
        ...

    def to_srt_strs(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> list[str]:
        """Return SRT format strings, one per file."""
        ...

    def to_srt(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> SRT:
        """Convert to an SRT object."""
        ...

    def to_srt_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        participants: Sequence[str] | None = None,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write SRT (.srt) files to a directory."""
        ...

    def to_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write TextGrid files to a directory."""
        ...

    def append(self, other: TextGrid, /) -> None:
        """Append data from another TextGrid reader."""
        ...

    def append_left(self, other: TextGrid, /) -> None:
        """Left-append data from another TextGrid reader, preserving order."""
        ...

    def extend(self, others: Sequence[TextGrid], /) -> None:
        """Extend data from multiple TextGrid readers."""
        ...

    def pop(self) -> TextGrid:
        """Remove and return the last file as a new TextGrid reader."""
        ...

    def pop_left(self) -> TextGrid:
        """Remove and return the first file as a new TextGrid reader."""
        ...

    def clear(self) -> None:
        """Remove all data from this reader."""
        ...

    def __add__(self, other: TextGrid) -> TextGrid: ...
    def __iadd__(self, other: TextGrid) -> TextGrid: ...
    def __iter__(self) -> Iterator[TextGrid]: ...
    def __getitem__(self, index: int | slice) -> TextGrid: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def read_textgrid(
    path: str | os.PathLike[str],
    *,
    cls: type[TextGrid] = TextGrid,
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
