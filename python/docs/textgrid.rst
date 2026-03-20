TextGrid (Praat)
================

The ``rustling.textgrid`` module provides tools for parsing
`Praat <https://www.fon.hum.uva.nl/praat/>`_ TextGrid annotation files.

A TextGrid file contains one or more tiers, each holding either
time-aligned intervals or time-stamped points:

.. code-block:: text

   File type = "ooTextFile"
   Object class = "TextGrid"

   xmin = 0
   xmax = 2.3
   tiers? <exists>
   size = 1
   item []:
       item [1]:
           class = "IntervalTier"
           name = "words"
           xmin = 0
           xmax = 2.3
           intervals: size = 2
               intervals [1]:
                   xmin = 0
                   xmax = 1.5
                   text = "hello"
               intervals [2]:
                   xmin = 1.5
                   xmax = 2.3
                   text = "world"

Both the normal "text" format and the compact "short text" format are supported.

Loading Data
------------

:func:`~rustling.read_textgrid`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quickest way to load TextGrid data is with :func:`~rustling.read_textgrid`.
It accepts a file path, directory, ZIP archive, git URL, or HTTP URL
and figures out the right loading strategy automatically:

.. code-block:: python

   import rustling

   # From a local .TextGrid file
   tg = rustling.read_textgrid("path/to/recording.TextGrid")

   # From a directory (recursively finds all .TextGrid files)
   tg = rustling.read_textgrid("path/to/corpus/")

   # From a ZIP archive
   tg = rustling.read_textgrid("path/to/corpus.zip")

   # From a git repository
   tg = rustling.read_textgrid("https://github.com/user/corpus.git")

   # From a URL (ZIP files are automatically detected and extracted)
   tg = rustling.read_textgrid("https://example.com/corpus.zip")

Using the class methods directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need finer control — for example, to pass specific files,
filter by regex, change the file extension, control caching, or parse
in-memory strings — use the :py:class:`~rustling.textgrid.TextGrid` class methods directly:

.. code-block:: python

   from rustling.textgrid import TextGrid

From specific files:

.. code-block:: python

   tg = TextGrid.from_files(["path/to/file1.TextGrid", "path/to/file2.TextGrid"])

From a directory with a regex filter:

.. code-block:: python

   tg = TextGrid.from_dir("path/to/corpus/", match=r"speaker_01")

The ``extension`` parameter controls which file extension to look for (default: ``".TextGrid"``).

From a ZIP archive:

.. code-block:: python

   tg = TextGrid.from_zip("path/to/corpus.zip")

From a git repository:

.. code-block:: python

   tg = TextGrid.from_git("https://github.com/user/corpus.git")

From a URL (ZIP files are automatically detected and extracted):

.. code-block:: python

   tg = TextGrid.from_url("https://example.com/corpus.zip")

From in-memory strings:

.. code-block:: python

   tg = TextGrid.from_strs([textgrid_string_1, textgrid_string_2])

Parallel processing
^^^^^^^^^^^^^^^^^^^

All loading methods accept a ``parallel`` parameter (default: ``True``)
to enable parallel parsing of multiple files.

Accessing Tiers and Annotations
-------------------------------

Each TextGrid file contains tiers that can be either interval tiers or point tiers.
Call :py:meth:`~rustling.textgrid.TextGrid.tiers` to get a list of lists,
one per file, where each inner list contains
:py:class:`~rustling.textgrid.IntervalTier` and/or
:py:class:`~rustling.textgrid.TextTier` objects:

.. code-block:: python

   import rustling
   from rustling.textgrid import IntervalTier, TextTier

   tg = rustling.read_textgrid("path/to/corpus/")

   for file_tiers in tg.tiers():
       for tier in file_tiers:
           print(tier.name, tier.tier_class)
           if isinstance(tier, IntervalTier):
               for interval in tier.intervals:
                   print(f"  [{interval.xmin}-{interval.xmax}] {interval.text}")
           elif isinstance(tier, TextTier):
               for point in tier.points:
                   print(f"  [{point.number}] {point.mark}")

An :py:class:`~rustling.textgrid.IntervalTier` has:

- ``name`` -- Tier name.
- ``xmin`` -- Start time in seconds.
- ``xmax`` -- End time in seconds.
- ``intervals`` -- List of :py:class:`~rustling.textgrid.Interval` objects.
- ``tier_class`` -- Always ``"IntervalTier"``.

An :py:class:`~rustling.textgrid.Interval` has:

- ``xmin`` -- Start time in seconds.
- ``xmax`` -- End time in seconds.
- ``text`` -- The annotation text.

A :py:class:`~rustling.textgrid.TextTier` has:

- ``name`` -- Tier name.
- ``xmin`` -- Start time in seconds.
- ``xmax`` -- End time in seconds.
- ``points`` -- List of :py:class:`~rustling.textgrid.Point` objects.
- ``tier_class`` -- Always ``"TextTier"``.

A :py:class:`~rustling.textgrid.Point` has:

- ``number`` -- Time in seconds.
- ``mark`` -- The annotation text.

Converting to ELAN
------------------

A :py:class:`~rustling.textgrid.TextGrid` reader can convert its data to
`ELAN <https://archive.mpi.nl/tla/elan>`_ format.

.. code-block:: python

   import rustling

   tg = rustling.read_textgrid("recording.TextGrid")

   # Convert to an ELAN object
   elan = tg.to_elan()

   # Or get EAF XML strings
   eaf_strs = tg.to_elan_strs()

   # Or write .eaf files directly
   tg.to_elan_files("output_dir/")

**Mapping:**

- Each IntervalTier becomes an ELAN tier with alignable annotations.
- TextTiers are skipped (point annotations have no duration for ELAN).
- Empty-text intervals are skipped.
- Times are converted from seconds to milliseconds.

Converting to CHAT
------------------

A :py:class:`~rustling.textgrid.TextGrid` reader can convert its data to CHAT format
for use with `CHILDES <https://childes.talkbank.org/>`_ / TalkBank tools.

.. code-block:: python

   import rustling

   tg = rustling.read_textgrid("recording.TextGrid")

   # Convert to a CHAT object
   chat = tg.to_chat()

   # Or get CHAT-formatted strings
   chat_strs = tg.to_chat_strs()

   # Or write .cha files directly
   tg.to_chat_files("output_dir/")

**Participant selection:**

By default, only IntervalTiers with a 3-character name are treated as
CHAT main tiers. To override this, pass the ``participants`` keyword argument:

.. code-block:: python

   chat = tg.to_chat(participants=["words", "phones"])

Converting to SRT
-----------------

A :py:class:`~rustling.textgrid.TextGrid` reader can convert its data to SRT
(SubRip Subtitle) format.

.. code-block:: python

   import rustling

   tg = rustling.read_textgrid("recording.TextGrid")

   # Convert to an SRT object
   srt = tg.to_srt()

   # Or get SRT-formatted strings
   srt_strs = tg.to_srt_strs()

   # Or write .srt files directly
   tg.to_srt_files("output_dir/")

**Participant selection** works the same as for CHAT conversion above.

Collection Operations
---------------------

A :py:class:`~rustling.textgrid.TextGrid` reader behaves like a collection of files.
You can iterate, slice, combine, and modify it:

.. code-block:: python

   import rustling

   tg = rustling.read_textgrid("path/to/corpus/")

   # File count and paths
   print(tg.n_files)
   print(tg.file_paths)

   # Iteration and slicing
   for single_file in tg:
       print(single_file.n_files)  # 1

   subset = tg[0:3]

   # Combining
   combined = tg1 + tg2
   tg1 += tg2

   # Appending and extending
   tg1.append(tg2)
   tg1.extend([tg2, tg3])

   # Removing
   last = tg.pop()
   first = tg.pop_left()
   tg.clear()
