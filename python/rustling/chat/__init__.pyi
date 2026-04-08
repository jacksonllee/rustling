"""CHAT data handling."""

from __future__ import annotations

import datetime
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, Sequence, overload

if TYPE_CHECKING:
    from rustling.conllu import CoNLLU
    from rustling.elan import ELAN
    from rustling.ngram import Ngrams
    from rustling.srt import SRT
    from rustling.textgrid import TextGrid

class Age:
    """Age in the CHAT format: years;months.days."""

    years: int
    """Number of years."""
    months: int | None
    """Number of months."""
    days: int | None
    """Number of days."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def in_months(self) -> float:
        """Return the age in total months as a float."""
        ...

class Participant:
    """A participant from @Participants and @ID headers."""

    code: str
    """Three-letter speaker ID (e.g., "CHI")."""
    name: str
    """Speaker name (may be empty)."""
    role: str
    """Standard role (e.g., "Target_Child")."""
    language: str | None
    """Language from @ID."""
    corpus: str | None
    """Corpus name from @ID."""
    age: Age | None
    """Age from @ID."""
    sex: str | None
    """Sex from @ID."""
    group: str | None
    """Group from @ID."""
    ses: str | None
    """Ethnicity/SES from @ID."""
    education: str | None
    """Education level from @ID."""
    custom: str | None
    """Custom field from @ID."""
    birth: str | None
    """Birth date from @Birth of header."""
    birthplace: str | None
    """Birthplace from @Birthplace of header."""
    l1: str | None
    """First language from @L1 of header."""

    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class Headers:
    """File-level headers from a CHAT file."""

    pid: str | None
    """Persistent identifier from @PID."""
    languages: list[str]
    """Language codes from @Languages."""
    participants: list[Participant]
    """Participants from @Participants and @ID."""
    options: str | None
    """Options from @Options."""
    media: dict[str, str | None] | None
    """Media descriptor from @Media as a dict with keys "filename", "format", and "status"."""
    date: datetime.date | None
    """Date from @Date, parsed as a date object.

    The CHAT format ``DD-MMM-YYYY`` (e.g., ``25-JAN-1983``) is tried first,
    then ISO ``YYYY-MM-DD``.  If neither format matches, the value is ``None``.
    """
    location: str | None
    """Location from @Location."""
    number: str | None
    """Number of participants from @Number."""
    recording_quality: str | None
    """Recording quality from @Recording Quality."""
    room_layout: str | None
    """Room layout from @Room Layout."""
    tape_location: str | None
    """Tape location from @Tape Location."""
    time_duration: str | None
    """Time duration from @Time Duration."""
    time_start: str | None
    """Time start from @Time Start."""
    transcriber: str | None
    """Transcriber from @Transcriber."""
    transcription: str | None
    """Transcription type from @Transcription."""
    types: str | None
    """Types from @Types."""
    videos: str | None
    """Videos from @Videos."""
    warning: str | None
    """Warning from @Warning."""
    situation: str | None
    """Situation from @Situation."""
    comments: list[str] | None
    """All @Comment values from the header section, in order."""
    other: dict[str, str]
    """Unrecognized headers as key-value pairs."""

    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class ChangeableHeader:
    """A changeable header that can appear mid-file in CHAT transcripts.

    Variants: Activities, Bck, Bg, Blank, Comment, Date, Eg,
    G, NewEpisode, Page, Situation.
    """

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

    class Activities:
        value: str
        def __init__(self, value: str) -> None: ...

    class Bck:
        value: str
        def __init__(self, value: str) -> None: ...

    class Bg:
        value: str | None
        def __init__(self, value: str | None = None) -> None: ...

    class Blank:
        def __init__(self) -> None: ...

    class Comment:
        value: str
        def __init__(self, value: str) -> None: ...

    class Date:
        value: str
        def __init__(self, value: str) -> None: ...

    class Eg:
        value: str | None
        def __init__(self, value: str | None = None) -> None: ...

    class G:
        value: str | None
        def __init__(self, value: str | None = None) -> None: ...

    class NewEpisode:
        def __init__(self) -> None: ...

    class Page:
        value: str
        def __init__(self, value: str) -> None: ...

    class Situation:
        value: str
        def __init__(self, value: str) -> None: ...

class Gra:
    """A grammatical relation from the %gra tier."""

    dep: int
    """Position of the dependent word."""
    head: int
    """Position of the head word."""
    rel: str
    """Grammatical relation type."""

    def __init__(self, dep: int, head: int, rel: str) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class Token:
    """A token with word, POS, morphology, and grammatical relation."""

    word: str
    """The word form."""
    pos: str | None
    """Part-of-speech tag from the %mor tier."""
    mor: str | None
    """Morphological information from the %mor tier."""
    gra: Gra | None
    """Grammatical relation from the %gra tier."""

    def __init__(
        self,
        word: str,
        pos: str | None = None,
        mor: str | None = None,
        gra: Gra | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class Utterance:
    """A single utterance from a CHAT transcript.

    For changeable headers (e.g., ``@Comment``, ``@New Episode``), only
    ``changeable_header`` is set; all other fields are ``None``.
    """

    participant: str | None
    """Speaker code (e.g., "CHI", "MOT"), or None for headers."""
    tokens: list[Token] | None
    """List of tokens in this utterance, or None for headers."""
    audible: str | None
    """Audibly faithful transcript of this utterance, or None for headers."""
    time_marks: tuple[int, int] | None
    """Start and end timestamps in milliseconds."""
    tiers: dict[str, str] | None
    """Raw tier data including the main tier and dependent tiers, or None for headers."""
    changeable_header: ChangeableHeader | None
    """The header variant if this is a changeable header, or None for real utterances."""
    mor_tier_name: str | None
    """The %-prefixed tier name used as the morphology tier (e.g., "%mor", "%xmor"),
    or None if mor+gra handling was disabled."""
    gra_tier_name: str | None
    """The %-prefixed tier name used as the grammatical relation tier (e.g., "%gra"),
    or None if mor+gra handling was disabled."""

    def __init__(
        self,
        *,
        participant: str | None = None,
        tokens: list[Token] | None = None,
        time_marks: tuple[int, int] | None = None,
        tiers: dict[str, str] | None = None,
        changeable_header: ChangeableHeader | None = None,
        mor_tier_name: str | None = "%mor",
        gra_tier_name: str | None = "%gra",
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def _repr_html_(self) -> str:
        """Return an HTML table representation for Jupyter notebooks."""
        ...

    def to_str(self) -> str:
        """Return a plain text tabular representation of this utterance.

        Returns:
            A column-aligned string with participant words, %mor, %gra,
            other tiers, and time marks. For changeable headers, returns
            the CHAT-format string (e.g., ``@Comment:\\tChild laughs``).
        """
        ...

class Utterances:
    """A sequence of utterances with formatted display.

    Returned by :meth:`CHAT.head` and :meth:`CHAT.tail`.
    Displays as column-aligned plain text in the terminal
    and as HTML tables in Jupyter notebooks.
    """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_html_(self) -> str: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> Utterance: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterator[Utterance]: ...

class CHAT:
    """CHAT data reader for CHILDES/TalkBank transcripts.

    This class parses CHAT transcription files and provides access
    to utterances, tokens, words, and annotations.
    """

    def __init__(self) -> None: ...
    @classmethod
    def from_strs(
        cls,
        strs: Sequence[str],
        ids: Sequence[str] | None = None,
        parallel: bool = True,
        strict: bool = True,
        mor_tier: str | None = "%mor",
        gra_tier: str | None = "%gra",
    ) -> CHAT:
        """Parse CHAT data from in-memory strings.

        Args:
            strs: CHAT-formatted strings to parse.
            ids: Optional identifiers for each string. If None, UUIDs
                are generated.
            parallel: If True, use parallel processing. Set to False
                to disable multithreading.
            strict: If True (default), raise ValueError on mor/word
                misalignment. If False, emit a warning and set tokens
                to an empty list for affected utterances.
            mor_tier: Name of the dependent tier to treat as the
                morphology tier, e.g. ``"%mor"`` or ``"%xmor"``.
                Set to None to disable mor+gra handling.
            gra_tier: Name of the dependent tier to treat as the
                grammatical relation tier, e.g. ``"%gra"`` or ``"%xgra"``.
                Set to None to disable mor+gra handling.

        Returns:
            A new CHAT reader with the parsed data.

        Raises:
            ValueError: If strs and ids have different lengths, or if
                strict is True and mor/word misalignment is found.
        """
        ...

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | os.PathLike[str]],
        *,
        parallel: bool = True,
        strict: bool = True,
        mor_tier: str | None = "%mor",
        gra_tier: str | None = "%gra",
    ) -> CHAT:
        """Load CHAT data from file paths.

        Args:
            paths: Paths to CHAT files.
            parallel: If True, use parallel processing. Set to False
                to disable multithreading.
            strict: If True (default), raise ValueError on mor/word
                misalignment. If False, emit a warning and set tokens
                to an empty list for affected utterances.
            mor_tier: Name of the dependent tier to treat as the
                morphology tier, e.g. ``"%mor"`` or ``"%xmor"``.
                Set to None to disable mor+gra handling.
            gra_tier: Name of the dependent tier to treat as the
                grammatical relation tier, e.g. ``"%gra"`` or ``"%xgra"``.
                Set to None to disable mor+gra handling.

        Returns:
            A new CHAT reader with the parsed data.

        Raises:
            ValueError: If strict is True and mor/word misalignment
                is found.
        """
        ...

    @classmethod
    def from_dir(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".cha",
        parallel: bool = True,
        strict: bool = True,
        mor_tier: str | None = "%mor",
        gra_tier: str | None = "%gra",
    ) -> CHAT:
        """Recursively load CHAT data from a directory.

        Args:
            path: Directory path to search.
            match: Regex pattern to include only matching file paths.
            extension: File extension to filter by (default: ".cha").
            parallel: If True, use parallel processing. Set to False
                to disable multithreading.
            strict: If True (default), raise ValueError on mor/word
                misalignment. If False, emit a warning and set tokens
                to an empty list for affected utterances.
            mor_tier: Name of the dependent tier to treat as the
                morphology tier, e.g. ``"%mor"`` or ``"%xmor"``.
                Set to None to disable mor+gra handling.
            gra_tier: Name of the dependent tier to treat as the
                grammatical relation tier, e.g. ``"%gra"`` or ``"%xgra"``.
                Set to None to disable mor+gra handling.

        Returns:
            A new CHAT reader with the parsed data.

        Raises:
            ValueError: If strict is True and mor/word misalignment
                is found.
        """
        ...

    @classmethod
    def from_zip(
        cls,
        path: str | os.PathLike[str],
        *,
        match: str | None = None,
        extension: str = ".cha",
        parallel: bool = True,
        strict: bool = True,
        mor_tier: str | None = "%mor",
        gra_tier: str | None = "%gra",
    ) -> CHAT:
        """Load CHAT data from a ZIP archive.

        Args:
            path: Path to the ZIP file.
            match: Regex pattern to include only matching file paths.
            extension: File extension to filter by (default: ".cha").
            parallel: If True, use parallel processing. Set to False
                to disable multithreading.
            strict: If True (default), raise ValueError on mor/word
                misalignment. If False, emit a warning and set tokens
                to an empty list for affected utterances.
            mor_tier: Name of the dependent tier to treat as the
                morphology tier, e.g. ``"%mor"`` or ``"%xmor"``.
                Set to None to disable mor+gra handling.
            gra_tier: Name of the dependent tier to treat as the
                grammatical relation tier, e.g. ``"%gra"`` or ``"%xgra"``.
                Set to None to disable mor+gra handling.

        Returns:
            A new CHAT reader with the parsed data.

        Raises:
            ValueError: If strict is True and mor/word misalignment
                is found.
        """
        ...

    @classmethod
    def from_git(
        cls,
        url: str,
        *,
        rev: str | None = None,
        depth: int | None = None,
        match: str | None = None,
        extension: str = ".cha",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
        strict: bool = True,
        mor_tier: str | None = "%mor",
        gra_tier: str | None = "%gra",
    ) -> CHAT:
        """Load CHAT data from a git repository.

        Clones the repository (or uses a cached clone) and parses all
        matching files from the resulting directory.

        Args:
            url: Git repository URL.
            rev: Branch, tag, or commit hash. If None, uses the
                repository's default branch.
            depth: Clone depth. Defaults to 1 (shallow clone).
                Ignored when rev is a commit hash.
            match: Regex pattern to include only matching file paths.
            extension: File extension to filter by (default: ".cha").
            cache_dir: Directory for caching cloned repositories.
                Defaults to ``~/.rustling/cache/``.
            force_download: If True, re-clone even if a cached copy exists.
            parallel: If True, use parallel processing.
            strict: If True (default), raise ValueError on mor/word
                misalignment. If False, emit a warning and set tokens
                to an empty list for affected utterances.
            mor_tier: Name of the dependent tier to treat as the
                morphology tier. Set to None to disable.
            gra_tier: Name of the dependent tier to treat as the
                grammatical relation tier. Set to None to disable.

        Returns:
            A new CHAT reader with the parsed data.
        """
        ...

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        match: str | None = None,
        extension: str = ".cha",
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        parallel: bool = True,
        strict: bool = True,
        mor_tier: str | None = "%mor",
        gra_tier: str | None = "%gra",
    ) -> CHAT:
        """Load CHAT data from a URL.

        Downloads the file (or uses a cached copy) and parses it.
        ZIP files are automatically detected and extracted.

        Args:
            url: URL to download from.
            match: Regex pattern to include only matching file paths
                (applicable for ZIP files).
            extension: File extension to filter by (default: ".cha",
                applicable for ZIP files).
            cache_dir: Directory for caching downloads.
                Defaults to ``~/.rustling/cache/``.
            force_download: If True, re-download even if a cached
                copy exists.
            parallel: If True, use parallel processing.
            strict: If True (default), raise ValueError on mor/word
                misalignment. If False, emit a warning and set tokens
                to an empty list for affected utterances.
            mor_tier: Name of the dependent tier to treat as the
                morphology tier. Set to None to disable.
            gra_tier: Name of the dependent tier to treat as the
                grammatical relation tier. Set to None to disable.

        Returns:
            A new CHAT reader with the parsed data.
        """
        ...

    @classmethod
    def from_utterances(
        cls,
        utterances: Sequence[Utterance],
    ) -> CHAT:
        """Construct a CHAT reader from a list of utterances.

        Creates a new reader containing a single virtual file with the given
        utterances. Useful for splitting a reader into sub-readers based on
        utterance boundaries. Raw lines are synthesized from each
        utterance's ``tiers`` data, so ``to_strs()`` and ``to_files()``
        produce valid CHAT output.

        Args:
            utterances: Utterance objects to include.

        Returns:
            A new CHAT reader containing the given utterances.
        """
        ...

    @property
    def file_paths(self) -> list[str]:
        """Return the list of file paths.

        Returns:
            File paths or identifiers.
        """
        ...

    @property
    def n_files(self) -> int:
        """Return the number of files.

        Returns:
            Number of loaded files.
        """
        ...

    def filter(
        self,
        *,
        files: str | Sequence[str] | None = None,
        participants: str | Sequence[str] | None = None,
    ) -> CHAT:
        """Return a new CHAT filtered by file path and/or participant regex.

        Args:
            files: Regex pattern(s) to include only matching file paths.
                Accepts a single string or a sequence of strings.
                Multiple patterns are OR'd.
            participants: Regex pattern(s) to include only matching
                participant codes. Accepts a single string or a sequence
                of strings. Patterns are auto-anchored (full match).
                Multiple patterns are OR'd.

        Returns:
            A new filtered CHAT reader.
        """
        ...

    def headers(self) -> list[Headers]:
        """Return file-level headers.

        Returns:
            A list of Headers, one per file.
        """
        ...

    @overload
    def participants(self, *, by_file: Literal[False] = False) -> list[Participant]: ...
    @overload
    def participants(self, *, by_file: Literal[True]) -> list[list[Participant]]: ...
    @overload
    def participants(
        self, *, by_file: bool = False
    ) -> list[Participant] | list[list[Participant]]: ...
    def participants(  # type: ignore[misc]
        self, *, by_file: bool = False
    ) -> list[Participant] | list[list[Participant]]:
        """Return participants.

        Args:
            by_file: If True, group participants by file.

        Returns:
            Participants, optionally grouped by file.
        """
        ...

    @overload
    def languages(self, *, by_file: Literal[False] = False) -> list[str]: ...
    @overload
    def languages(self, *, by_file: Literal[True]) -> list[list[str]]: ...
    @overload
    def languages(self, *, by_file: bool = False) -> list[str] | list[list[str]]: ...
    def languages(  # type: ignore[misc]
        self, *, by_file: bool = False
    ) -> list[str] | list[list[str]]:
        """Return languages.

        Args:
            by_file: If True, group languages by file.

        Returns:
            Language codes, optionally grouped by file.
        """
        ...

    @overload
    def utterances(self, *, by_file: Literal[False] = False) -> list[Utterance]: ...
    @overload
    def utterances(self, *, by_file: Literal[True]) -> list[list[Utterance]]: ...
    @overload
    def utterances(
        self, *, by_file: bool = False
    ) -> list[Utterance] | list[list[Utterance]]: ...
    def utterances(  # type: ignore[misc]
        self, *, by_file: bool = False
    ) -> list[Utterance] | list[list[Utterance]]:
        """Return utterances.

        Args:
            by_file: If True, group utterances by file.

        Returns:
            Utterances, optionally grouped by file.
        """
        ...

    def head(self, n: int = 5) -> Utterances:
        """Return the first n utterances with a formatted display.

        Args:
            n: Number of utterances to include.

        Returns:
            An Utterances object that displays as formatted text.
        """
        ...

    def tail(self, n: int = 5) -> Utterances:
        """Return the last n utterances with a formatted display.

        Args:
            n: Number of utterances to include.

        Returns:
            An Utterances object that displays as formatted text.
        """
        ...

    @overload
    def words(
        self,
        *,
        by_utterance: Literal[False] = False,
        by_file: Literal[False] = False,
    ) -> list[str]: ...
    @overload
    def words(
        self,
        *,
        by_utterance: Literal[True],
        by_file: Literal[False] = False,
    ) -> list[list[str]]: ...
    @overload
    def words(
        self,
        *,
        by_utterance: Literal[False] = False,
        by_file: Literal[True],
    ) -> list[list[str]]: ...
    @overload
    def words(
        self,
        *,
        by_utterance: Literal[True],
        by_file: Literal[True],
    ) -> list[list[list[str]]]: ...
    @overload
    def words(
        self, *, by_utterance: bool = False, by_file: bool = False
    ) -> list[str] | list[list[str]] | list[list[list[str]]]: ...
    def words(  # type: ignore[misc]
        self, *, by_utterance: bool = False, by_file: bool = False
    ) -> list[str] | list[list[str]] | list[list[list[str]]]:
        """Return words.

        Args:
            by_utterance: If True, group words by utterance.
            by_file: If True, group words by file.

        Returns:
            Words with optional grouping.
        """
        ...

    @overload
    def tokens(
        self,
        *,
        by_utterance: Literal[False] = False,
        by_file: Literal[False] = False,
    ) -> list[Token]: ...
    @overload
    def tokens(
        self,
        *,
        by_utterance: Literal[True],
        by_file: Literal[False] = False,
    ) -> list[list[Token]]: ...
    @overload
    def tokens(
        self,
        *,
        by_utterance: Literal[False] = False,
        by_file: Literal[True],
    ) -> list[list[Token]]: ...
    @overload
    def tokens(
        self,
        *,
        by_utterance: Literal[True],
        by_file: Literal[True],
    ) -> list[list[list[Token]]]: ...
    @overload
    def tokens(
        self, *, by_utterance: bool = False, by_file: bool = False
    ) -> list[Token] | list[list[Token]] | list[list[list[Token]]]: ...
    def tokens(  # type: ignore[misc]
        self, *, by_utterance: bool = False, by_file: bool = False
    ) -> list[Token] | list[list[Token]] | list[list[list[Token]]]:
        """Return tokens.

        Args:
            by_utterance: If True, group tokens by utterance.
            by_file: If True, group tokens by file.

        Returns:
            Tokens with optional grouping.
        """
        ...

    def mlum(
        self,
        *,
        participant: str = "CHI",
        n: int | None = 100,
    ) -> list[float]:
        """Mean length of utterance in morphemes.

        Args:
            participant: Target participant code.
            n: Number of utterances to use per file. None for all.

        Returns:
            One value per file.
        """
        ...

    def mlu(
        self,
        *,
        participant: str = "CHI",
        n: int | None = 100,
    ) -> list[float]:
        """Mean length of utterance in morphemes.

        Alias for :meth:`mlum`.

        Args:
            participant: Target participant code.
            n: Number of utterances to use per file. None for all.

        Returns:
            One value per file.
        """
        ...

    def mluw(
        self,
        *,
        participant: str = "CHI",
        n: int | None = 100,
    ) -> list[float]:
        """Mean length of utterance in words.

        Args:
            participant: Target participant code.
            n: Number of utterances to use per file. None for all.

        Returns:
            One value per file.
        """
        ...

    def ttr(
        self,
        *,
        participant: str = "CHI",
        n: int | None = 350,
    ) -> list[float]:
        """Type-token ratio for non-punctuation words.

        Args:
            participant: Target participant code.
            n: Number of tokens to use per file. None for all.

        Returns:
            One value per file.
        """
        ...

    def ipsyn(
        self,
        *,
        participant: str = "CHI",
        n: int | None = 100,
    ) -> list[int]:
        """Index of Productive Syntax (IPSyn).

        Args:
            participant: Target participant code.
            n: Number of utterances to use per file. None for all.

        Returns:
            One score (0-112) per file.
        """
        ...

    def ages(self) -> list[Age | None]:
        """Return the age of the target child (CHI) in each file.

        Returns:
            One Age per file, or None if the file has no CHI or the CHI
            has no age.
        """
        ...

    def word_ngrams(self, n: int) -> Ngrams:
        """Return an Ngrams for word n-grams across all utterances.

        N-grams do not cross utterance boundaries.

        Args:
            n: The n-gram order (1 for unigrams, 2 for bigrams, etc.).

        Returns:
            An Ngrams with the accumulated counts.

        Raises:
            ValueError: If n < 1.
        """
        ...

    def append(self, other: CHAT, /) -> None:
        """Append data from another CHAT reader.

        Args:
            other: A CHAT reader whose data to append.
        """
        ...

    def append_left(self, other: CHAT, /) -> None:
        """Left-append data from another CHAT reader.

        Args:
            other: A CHAT reader whose data to prepend.
        """
        ...

    def extend(self, others: Sequence[CHAT], /) -> None:
        """Extend data from multiple CHAT readers.

        Args:
            others: CHAT readers whose data to append.
        """
        ...

    def extend_left(self, others: Sequence[CHAT], /) -> None:
        """Left-extend data from multiple CHAT readers.

        Args:
            others: CHAT readers whose data to prepend.
        """
        ...

    def pop(self) -> CHAT:
        """Remove and return the last file as a new CHAT reader.

        Returns:
            A new CHAT reader containing the removed file.

        Raises:
            IndexError: If the reader is empty.
        """
        ...

    def pop_left(self) -> CHAT:
        """Remove and return the first file as a new CHAT reader.

        Returns:
            A new CHAT reader containing the removed file.

        Raises:
            IndexError: If the reader is empty.
        """
        ...

    def clear(self) -> None:
        """Remove all data from this reader."""
        ...

    def to_strs(self) -> list[str]:
        """Return CHAT data strings, one per file.

        Returns:
            A list of CHAT-formatted strings.
        """
        ...

    def to_files(
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
                If None, filenames are derived from the original source
                file paths (e.g., ``foo.cha`` stays ``foo.cha``). Falls
                back to ``0001.cha``, ``0002.cha``, etc. when the data
                was parsed from in-memory strings.

        Raises:
            ValueError: If filenames count doesn't match file count.
            IOError: If writing fails.
        """
        ...

    def to_elan_strs(self) -> list[str]:
        """Return EAF XML strings, one per file.

        Converts the CHAT data to ELAN XML format. Each CHAT file
        produces one EAF XML string. Participants become alignable
        tiers, and dependent tiers (e.g., %mor, %gra) become
        reference annotation tiers named ``{tier}@{participant}``
        (e.g., ``mor@CHI``).

        Returns:
            A list of EAF XML strings.
        """
        ...

    def to_elan(self) -> ELAN:
        """Convert to an ELAN object.

        Each CHAT file produces one ELAN file. Participants become
        alignable tiers, and dependent tiers (e.g., %mor, %gra) become
        reference annotation tiers named ``{tier}@{participant}``
        (e.g., ``mor@CHI``).

        Returns:
            An :class:`~rustling.elan.ELAN` object.
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

        Converts the CHAT data to ELAN XML format and writes .eaf files.
        Each CHAT file produces one .eaf file. Participants become
        alignable tiers, and dependent tiers (e.g., %mor, %gra) become
        reference annotation tiers named ``{tier}@{participant}``
        (e.g., ``mor@CHI``).

        Args:
            dir_path: Directory path to write .eaf files to.
            filenames: Custom filenames for the output files.
                If None, filenames are derived from the original source
                file paths with the extension changed to ``.eaf``
                (e.g., ``foo.cha`` becomes ``foo.eaf``). Falls back to
                ``0001.eaf``, ``0002.eaf``, etc. when the data was
                parsed from in-memory strings.

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
            participants: Participant codes to include.
                If None, all participants are included.
                Utterances without time marks are skipped.

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

        Each CHAT file produces one SRT file. When multiple participants
        are present, subtitle text is prefixed with the participant code
        (e.g., ``"CHI: more cookie ."``). Utterances without time marks
        are skipped.

        Args:
            participants: Participant codes to include.
                If None, all participants are included.

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
            participants: Participant codes to include.
                If None, all participants are included.
            filenames: Custom filenames for the output files.
                If None, filenames are derived from the original source
                file paths with the extension changed to ``.srt``.

        Raises:
            ValueError: If filenames count doesn't match file count.
            IOError: If writing fails.
        """
        ...

    def to_textgrid_strs(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> list[str]:
        """Return TextGrid format strings, one per file."""
        ...

    def to_textgrid(
        self,
        *,
        participants: Sequence[str] | None = None,
    ) -> TextGrid:
        """Convert to a TextGrid object."""
        ...

    def to_textgrid_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        participants: Sequence[str] | None = None,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write TextGrid (.TextGrid) files to a directory."""
        ...

    def to_conllu_strs(self) -> list[str]:
        """Return CoNLL-U format strings, one per file."""
        ...

    def to_conllu(self) -> CoNLLU:
        """Convert to a CoNLL-U object.

        Returns:
            A :class:`~rustling.conllu.CoNLLU` object.
        """
        ...

    def to_conllu_files(
        self,
        dir_path: str | os.PathLike[str],
        /,
        *,
        filenames: Sequence[str] | None = None,
    ) -> None:
        """Write CoNLL-U (.conllu) files to a directory.

        Args:
            dir_path: Directory path to write .conllu files to.
            filenames: Custom filenames for the output files.

        Raises:
            ValueError: If filenames count doesn't match file count.
            IOError: If writing fails.
        """
        ...

    def info(self, *, verbose: bool = False) -> None:
        """Print a summary of this reader's data.

        Args:
            verbose: If True, show the details of all files.
                Defaults to False (shows first 5 files only).
        """
        ...

    def __add__(self, other: CHAT, /) -> CHAT: ...
    def __iadd__(self, other: CHAT, /) -> CHAT: ...
    @overload
    def __getitem__(self, index: int, /) -> CHAT: ...
    @overload
    def __getitem__(self, index: slice, /) -> CHAT: ...
    def __iter__(self) -> Iterator[CHAT]: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

def read_chat(
    path: str | os.PathLike[str],
    *,
    filter_files: str | Sequence[str] | None = None,
    filter_participants: str | Sequence[str] | None = None,
    cls: type[CHAT] = CHAT,
    strict: bool = True,
) -> CHAT:
    """Read CHAT data.

    Args:
        path: Path to a ``.zip`` file, a local directory containing ``.cha``
            files, a single ``.cha`` file, a git repository URL
            (ending in ``.git``), or an HTTP/HTTPS URL.
        filter_files: Filename(s) to keep.
            Regular expression matching is supported.
            If ``None``, all files are included.
        filter_participants: Participant code(s) to keep.
            Regular expression matching is supported.
            If ``None``, all participants are included.
        cls: The class used to create the reader. Must be ``CHAT`` or a
            subclass of it.
        strict: If ``True``, enforce strict parsing of the CHAT data.

    Returns:
        A ``CHAT`` instance filtered by the specified files and participants.

    Raises:
        TypeError: If *cls* is not ``CHAT`` or a subclass of it.
        ValueError: If *path* does not point to a recognized source.
    """
    ...

__all__ = [
    "Age",
    "CHAT",
    "ChangeableHeader",
    "Gra",
    "Headers",
    "Participant",
    "Token",
    "Utterance",
    "Utterances",
    "read_chat",
]
