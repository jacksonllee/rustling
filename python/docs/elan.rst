ELAN Parsing
============

The ``rustling.elan`` module provides tools for parsing
`ELAN <https://archive.mpi.nl/tla/elan>`_ annotation files (``.eaf``).

Loading Data
------------

:func:`~rustling.read_elan`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quickest way to load ELAN data is with :func:`~rustling.read_elan`.
It accepts a file path, directory, ZIP archive, git URL, or HTTP URL
and figures out the right loading strategy automatically:

.. code-block:: python

   import rustling

   # From a local .eaf file
   elan = rustling.read_elan("path/to/recording.eaf")

   # From a directory (recursively finds all .eaf files)
   elan = rustling.read_elan("path/to/corpus/")

   # From a ZIP archive
   elan = rustling.read_elan("path/to/corpus.zip")

   # From a git repository
   elan = rustling.read_elan("https://github.com/user/corpus.git")

   # From a URL (ZIP files are automatically detected and extracted)
   elan = rustling.read_elan("https://example.com/corpus.zip")

Using the class methods directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need finer control — for example, to pass specific files,
filter by regex, change the file extension, control caching, or parse
in-memory strings — use the :py:class:`~rustling.elan.ELAN` class methods directly:

.. code-block:: python

   from rustling.elan import ELAN

From specific files:

.. code-block:: python

   elan = ELAN.from_files(["path/to/file1.eaf", "path/to/file2.eaf"])

From a directory with a regex filter:

.. code-block:: python

   elan = ELAN.from_dir("path/to/corpus/", match=r"speaker_01")

The ``extension`` parameter controls which file extension to look for (default: ``".eaf"``).

From a ZIP archive:

.. code-block:: python

   elan = ELAN.from_zip("path/to/corpus.zip")

From a git repository:

.. code-block:: python

   elan = ELAN.from_git("https://github.com/user/corpus.git")

From a URL (ZIP files are automatically detected and extracted):

.. code-block:: python

   elan = ELAN.from_url("https://example.com/corpus.zip")

From in-memory strings:

.. code-block:: python

   elan = ELAN.from_strs([eaf_string_1, eaf_string_2])

Parallel processing
^^^^^^^^^^^^^^^^^^^

All loading methods accept a ``parallel`` parameter (default: ``True``)
to enable parallel parsing of multiple files.

Accessing Tiers and Annotations
-------------------------------

Each ELAN file contains annotation tiers.
Call :py:meth:`~rustling.elan.ELAN.tiers` to get a list of ``OrderedDict[str, Tier]``,
one per file:

.. code-block:: python

   import rustling

   elan = rustling.read_elan("path/to/corpus/")

   for file_tiers in elan.tiers():
       for tier_id, tier in file_tiers.items():
           print(tier_id, tier.participant, tier.linguistic_type_ref)
           for annotation in tier.annotations:
               print(f"  [{annotation.start_time}-{annotation.end_time}] {annotation.value}")

A :py:class:`~rustling.elan.Tier` has the following properties:

- ``id`` -- Tier ID (e.g., ``"G-jyutping"``).
- ``participant`` -- Participant name.
- ``annotator`` -- Annotator name.
- ``linguistic_type_ref`` -- Linguistic type reference.
- ``parent_id`` -- Parent tier ID, or ``None`` for root tiers.
- ``child_ids`` -- Child tier IDs, or ``None`` if no children.
- ``annotations`` -- List of :py:class:`~rustling.elan.Annotation` objects.

An :py:class:`~rustling.elan.Annotation` has:

- ``id`` -- Annotation ID (e.g., ``"a1"``).
- ``start_time`` -- Start time in milliseconds, or ``None`` if unresolvable.
- ``end_time`` -- End time in milliseconds, or ``None`` if unresolvable.
- ``value`` -- The annotation text content.
- ``parent_id`` -- Parent annotation ID for ``REF_ANNOTATION`` types, or ``None``.

Converting to CHAT
------------------

An :py:class:`~rustling.elan.ELAN` reader can convert its data to CHAT format
for use with `CHILDES <https://childes.talkbank.org/>`_ / TalkBank tools.

.. code-block:: python

   import rustling

   elan = rustling.read_elan("recording.eaf")

   # Convert to a CHAT object
   chat = elan.to_chat()

   # Or get CHAT-formatted strings
   chat_strs = elan.to_chat_strs()

   # Or write .cha files directly
   elan.to_chat_files("output_dir/")

**Tier mapping:**

- Parent (alignable) tiers become CHAT main tiers (e.g., ``*CHI:``).
- Child tiers whose ID matches ``{name}@{code}`` (e.g., ``mor@CHI``)
  become CHAT dependent tiers (e.g., ``%mor:``).
- ELAN ``Tier.participant`` populates the CHAT ``@Participants`` line.

**Participant selection:**

By default, only parent tiers with a 3-character ID are treated as
CHAT main tiers (matching the standard CHAT convention of 3-letter
participant codes like ``CHI``, ``MOT``, ``FAT``).
To override this, pass the ``participants`` keyword argument:

.. code-block:: python

   # Use specific tier IDs as CHAT main tiers
   chat = elan.to_chat(participants=["Speaker1", "Speaker2"])

   # Also works with to_chat_strs and to_chat_files
   elan.to_chat_files("output_dir/", participants=["Speaker1", "Speaker2"])

Converting to SRT
-----------------

An :py:class:`~rustling.elan.ELAN` reader can convert its data to SRT
(SubRip Subtitle) format.

.. code-block:: python

   import rustling

   elan = rustling.read_elan("recording.eaf")

   # Convert to an SRT object
   srt = elan.to_srt()

   # Or get SRT-formatted strings
   srt_strs = elan.to_srt_strs()

   # Or write .srt files directly
   elan.to_srt_files("output_dir/")

**Mapping:**

- Each selected annotation with time marks becomes one subtitle block.
- Annotations without time marks are skipped (SRT requires time ranges).
- When multiple tiers are selected, the subtitle text is prefixed with
  the tier ID (e.g., ``"CHI: more cookie ."``).
  For a single tier, no prefix is added.

**Participant selection:**

By default, only parent tiers with a 3-character ID are included
(matching the standard CHAT convention). To override this, pass
the ``participants`` keyword argument:

.. code-block:: python

   # Use specific tier IDs
   srt = elan.to_srt(participants=["Speaker1", "Speaker2"])

   # Also works with to_srt_strs and to_srt_files
   elan.to_srt_files("output_dir/", participants=["Speaker1", "Speaker2"])

Converting to TextGrid
----------------------

An :py:class:`~rustling.elan.ELAN` reader can convert its data to
`TextGrid <https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html>`_
format for use with Praat.

.. code-block:: python

   import rustling

   elan = rustling.read_elan("recording.eaf")

   # Convert to a TextGrid object
   textgrid = elan.to_textgrid()

   # Or get TextGrid-formatted strings
   textgrid_strs = elan.to_textgrid_strs()

   # Or write .TextGrid files directly
   elan.to_textgrid_files("output_dir/")

**Mapping:**

- Each ELAN tier becomes an IntervalTier.
- Annotations without time marks are skipped.
- Times are converted from milliseconds to seconds.

Collection Operations
---------------------

An :py:class:`~rustling.elan.ELAN` reader behaves like a collection of files.
You can iterate, slice, combine, and modify it:

.. code-block:: python

   import rustling

   elan = rustling.read_elan("path/to/corpus/")

   # File count and paths
   print(elan.n_files)
   print(elan.file_paths)

   # Iteration and slicing
   for single_file in elan:
       print(single_file.n_files)  # 1

   subset = elan[0:3]

   # Combining
   combined = elan1 + elan2
   elan1 += elan2

   # Appending and extending
   elan1.append(elan2)
   elan1.extend([elan2, elan3])

   # Removing
   last = elan.pop()
   first = elan.pop_left()
   elan.clear()
