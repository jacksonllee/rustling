"""ELAN (.eaf) file parsing."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from rustling.chat import CHAT
    from rustling.srt import SRT
    from rustling.textgrid import TextGrid

class Annotation:
    """A single annotation within an ELAN tier."""

    @property
    def id(self) -> str:
        """Annotation ID (e.g. "a1")."""
        ...

    @property
    def start_time(self) -> int | None:
        """Start time in milliseconds, or None if unresolvable."""
        ...

    @property
    def end_time(self) -> int | None:
        """End time in milliseconds, or None if unresolvable."""
        ...

    @property
    def value(self) -> str:
        """The annotation text content."""
        ...

    @property
    def parent_id(self) -> str | None:
        """Parent annotation ID for REF_ANNOTATIONs, or None."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class Tier:
    """An annotation tier (layer) within an ELAN file."""

    @property
    def id(self) -> str:
        """Tier ID (e.g. "G-jyutping")."""
        ...

    @property
    def participant(self) -> str:
        """Participant name."""
        ...

    @property
    def annotator(self) -> str:
        """Annotator name."""
        ...

    @property
    def linguistic_type_ref(self) -> str:
        """Linguistic type reference."""
        ...

    @property
    def parent_id(self) -> str | None:
        """Parent tier ID, or None for root tiers."""
        ...

    @property
    def child_ids(self) -> list[str] | None:
        """Child tier IDs, or None if no children."""
        ...

    @property
    def annotations(self) -> list[Annotation]:
        """Annotations in this tier."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class ELAN:
    """ELAN (.eaf) data reader."""

    def __init__(self) -> None: ...
    @classmethod
    def from_strs(
        cls,
        strs: Sequence[str],
        ids: Sequence[str] | None = None,
        parallel: bool = True,
    ) -> ELAN:
        """Parse ELAN data from in-memory strings."""
        ...

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | os.PathLike[str]],
        *,
        parallel: bool = True,
    ) -> ELAN:
        """Load ELAN data from file paths."""
        ...

    @classmethod
    def from_dir(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".eaf",
        parallel: bool = True,
    ) -> ELAN:
        """Recursively load ELAN data from a directory."""
        ...

    @classmethod
    def from_zip(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".eaf",
        parallel: bool = True,
    ) -> ELAN:
        """Load ELAN data from a ZIP archive."""
        ...

    @classmethod
    def from_git(
        cls,
        url: str,
        *,
        rev: str | None = None,
        depth: int | None = None,
        match: str | None = None,
        extension: str = ".eaf",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
    ) -> ELAN:
        """Load ELAN data from a git repository.

        Clones the repository (or uses a cached clone) and parses all
        matching files from the resulting directory.

        Args:
            url: Git repository URL.
            rev: Branch, tag, or commit hash. If None, uses the
                repository's default branch.
            depth: Clone depth. Defaults to 1 (shallow clone).
                Ignored when rev is a commit hash.
            match: Regex pattern to include only matching file paths.
            extension: File extension to filter by (default: ".eaf").
            cache_dir: Directory for caching cloned repositories.
                Defaults to ``~/.rustling/cache/``.
            force_download: If True, re-clone even if a cached copy exists.
            parallel: If True, use parallel processing.

        Returns:
            A new ELAN reader with the parsed data.
        """
        ...

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        match: str | None = None,
        extension: str = ".eaf",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
    ) -> ELAN:
        """Load ELAN data from a URL.

        Downloads the file (or uses a cached copy) and parses it.
        ZIP files are automatically detected and extracted.

        Args:
            url: URL to download from.
            match: Regex pattern to include only matching file paths
                (applicable for ZIP files).
            extension: File extension to filter by (default: ".eaf",
                applicable for ZIP files).
            cache_dir: Directory for caching downloads.
                Defaults to ``~/.rustling/cache/``.
            force_download: If True, re-download even if a cached
                copy exists.
            parallel: If True, use parallel processing.

        Returns:
            A new ELAN reader with the parsed data.
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

    def tiers(self) -> list[OrderedDict[str, Tier]]:
        """Return tiers as a list of OrderedDicts, one per file."""
        ...

    def to_strs(self) -> list[str]:
        """Return EAF XML strings, one per file."""
        ...

    def to_chat_strs(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> list[str]:
        """Return CHAT format strings, one per file.

        Args:
            participants: Participant codes (tier IDs) to treat as
                CHAT main tiers. If None, auto-detects parent tiers
                with a 3-character ID.

        Returns:
            A list of CHAT-formatted strings.
        """
        ...

    def to_chat(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> CHAT:
        """Convert to a CHAT object.

        Each ELAN file produces one CHAT file. Parent tiers become
        CHAT main tiers, and child tiers matching ``{name}@{code}``
        become dependent tiers (e.g., ``mor@CHI`` becomes ``%mor``).

        Args:
            participants: Participant codes (tier IDs) to treat as
                CHAT main tiers. If None, auto-detects parent tiers
                with a 3-character ID.

        Returns:
            A :class:`~rustling.chat.CHAT` object.
        """
        ...

    def to_chat_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        participants: Sequence[str] | None = None,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write CHAT (.cha) files to a directory.

        Args:
            dir_path: Directory path to write .cha files to.
            participants: Participant codes (tier IDs) to treat as
                CHAT main tiers. If None, auto-detects parent tiers
                with a 3-character ID.
            filenames: Custom filenames for the output files.
                If None, filenames are derived from the original source
                file paths with the extension changed to ``.cha``.
                Falls back to ``0001.cha``, ``0002.cha``, etc. when the
                data was parsed from in-memory strings.

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
        """Write ELAN (.eaf) files to a directory.

        Args:
            dir_path: Directory path to write .eaf files to.
            filenames: Custom filenames for the output files.
                If None, filenames are derived from the original source
                file paths. Falls back to ``0001.eaf``, ``0002.eaf``,
                etc. when the data was parsed from in-memory strings.

        Raises:
            ValueError: If filenames count doesn't match file count.
            IOError: If writing fails.
        """
        ...

    def to_srt_strs(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> list[str]:
        """Return SRT format strings, one per file.

        Args:
            participants: Participant codes (tier IDs) to include.
                If None, auto-detects parent tiers with a 3-character ID.
                Annotations without time marks are skipped.

        Returns:
            A list of SRT-formatted strings.
        """
        ...

    def to_srt(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> SRT:
        """Convert to an SRT object.

        Each ELAN file produces one SRT file. When multiple tiers are
        selected, subtitle text is prefixed with the tier ID
        (e.g., ``"CHI: more cookie ."``). Annotations without time marks
        are skipped.

        Args:
            participants: Participant codes (tier IDs) to include.
                If None, auto-detects parent tiers with a 3-character ID.

        Returns:
            A :class:`~rustling.srt.SRT` object.
        """
        ...

    def to_srt_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        participants: Sequence[str] | None = None,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write SRT (.srt) files to a directory.

        Args:
            dir_path: Directory path to write .srt files to.
            participants: Participant codes (tier IDs) to include.
                If None, auto-detects parent tiers with a 3-character ID.
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

    def append(self, other: ELAN, /) -> None:
        """Append data from another ELAN reader."""
        ...

    def append_left(self, other: ELAN, /) -> None:
        """Left-append data from another ELAN reader, preserving order."""
        ...

    def extend(self, others: Sequence[ELAN], /) -> None:
        """Extend data from multiple ELAN readers."""
        ...

    def pop(self) -> ELAN:
        """Remove and return the last file as a new ELAN reader."""
        ...

    def pop_left(self) -> ELAN:
        """Remove and return the first file as a new ELAN reader."""
        ...

    def clear(self) -> None:
        """Remove all data from this reader."""
        ...

    def __add__(self, other: ELAN, /) -> ELAN: ...
    def __iadd__(self, other: ELAN, /) -> ELAN: ...
    def __iter__(self) -> Iterator[ELAN]: ...
    def __getitem__(self, index: int | slice, /) -> ELAN: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

def read_elan(
    path: str | os.PathLike[str],
    *,
    cls: type[ELAN] = ELAN,
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
