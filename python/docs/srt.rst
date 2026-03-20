SRT (SubRip Subtitle)
=====================

The ``rustling.srt`` module provides tools for parsing
`SubRip <https://en.wikipedia.org/wiki/SubRip>`_ subtitle (``.srt``) files.

An ``.srt`` file is a plain-text format where each subtitle block has a
sequence number, a time range, and one or more lines of text:

.. code-block:: text

   1
   00:02:16,612 --> 00:02:19,376
   Senator, we're making
   our final approach into Coruscant.

   2
   00:02:19,482 --> 00:02:21,609
   Very good, Lieutenant.

Loading Data
------------

:func:`~rustling.read_srt`
^^^^^^^^^^^^^^^^^^^^^^^^^^

The quickest way to load SRT data is with :func:`~rustling.read_srt`.
It accepts a file path, directory, ZIP archive, git URL, or HTTP URL
and figures out the right loading strategy automatically:

.. code-block:: python

   import rustling

   # From a local .srt file
   srt = rustling.read_srt("path/to/movie.srt")

   # From a directory (recursively finds all .srt files)
   srt = rustling.read_srt("path/to/subtitles/")

   # From a ZIP archive
   srt = rustling.read_srt("path/to/subtitles.zip")

   # From a git repository
   srt = rustling.read_srt("https://github.com/user/corpus.git")

   # From a URL (ZIP files are automatically detected and extracted)
   srt = rustling.read_srt("https://example.com/subtitles.zip")

Using the class methods directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need finer control — for example, to pass specific files,
filter by regex, change the file extension, control caching, or parse
in-memory strings — use the :py:class:`~rustling.srt.SRT` class methods directly:

.. code-block:: python

   from rustling.srt import SRT

From specific files:

.. code-block:: python

   srt = SRT.from_files(["path/to/file1.srt", "path/to/file2.srt"])

From a directory with a regex filter:

.. code-block:: python

   srt = SRT.from_dir("path/to/subtitles/", match=r"episode_01")

The ``extension`` parameter controls which file extension to look for (default: ``".srt"``).

From a ZIP archive:

.. code-block:: python

   srt = SRT.from_zip("path/to/subtitles.zip")

From a git repository:

.. code-block:: python

   srt = SRT.from_git("https://github.com/user/corpus.git")

From a URL (ZIP files are automatically detected and extracted):

.. code-block:: python

   srt = SRT.from_url("https://example.com/subtitles.zip")

From in-memory strings:

.. code-block:: python

   srt = SRT.from_strs([srt_string_1, srt_string_2])

Parallel processing
^^^^^^^^^^^^^^^^^^^

All loading methods accept a ``parallel`` parameter (default: ``True``)
to enable parallel parsing of multiple files.

Accessing Subtitle Data
-----------------------

Call :py:meth:`~rustling.srt.SRT.utterances` to get a flat list of all
subtitle blocks across all files:

.. code-block:: python

   import rustling

   srt = rustling.read_srt("movie.srt")

   for utterance in srt.utterances():
       print(utterance.index, utterance.time_marks, utterance.line)

An :py:class:`~rustling.srt.Utterance` has the following properties:

- ``index`` -- 1-based sequence number from the SRT file.
- ``line`` -- The subtitle text (multiline text preserved with ``\n``).
- ``time_marks`` -- Start and end time in milliseconds as a ``tuple[int, int]``.

:py:class:`~rustling.srt.Utterance` objects can also be constructed directly:

.. code-block:: python

   from rustling.srt import Utterance

   utt = Utterance(index=1, line="Hello world.", time_marks=(0, 1500))

Converting to CHAT
------------------

An :py:class:`~rustling.srt.SRT` reader can convert its data to CHAT format
for use with `CHILDES <https://childes.talkbank.org/>`_ / TalkBank tools.

.. code-block:: python

   import rustling

   srt = rustling.read_srt("recording.srt")

   # Convert to a CHAT object
   chat = srt.to_chat()

   # Or get CHAT-formatted strings
   chat_strs = srt.to_chat_strs()

   # Or write .cha files directly
   srt.to_chat_files("output_dir/")

Since SRT files have no participant information, a default participant code
``"SPK"`` (Speaker) is used. Multiline subtitle text is joined with a space
in the CHAT output (CHAT utterances are single-line).

Converting to ELAN
------------------

An :py:class:`~rustling.srt.SRT` reader can convert its data to ELAN format.

.. code-block:: python

   import rustling

   srt = rustling.read_srt("recording.srt")

   # Convert to an ELAN object
   elan = srt.to_elan()

   # Or get EAF XML strings
   eaf_strs = srt.to_elan_strs()

   # Or write .eaf files directly
   srt.to_elan_files("output_dir/")

The conversion creates a single alignable tier named ``"SPK"`` (Speaker)
with one annotation per subtitle block.

Converting to TextGrid
----------------------

An :py:class:`~rustling.srt.SRT` reader can convert its data to
`TextGrid <https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html>`_
format for use with Praat.

.. code-block:: python

   import rustling

   srt = rustling.read_srt("recording.srt")

   # Convert to a TextGrid object
   textgrid = srt.to_textgrid()

   # Or get TextGrid-formatted strings
   textgrid_strs = srt.to_textgrid_strs()

   # Or write .TextGrid files directly
   srt.to_textgrid_files("output_dir/")

The conversion creates a single IntervalTier named ``"SPK"`` (Speaker)
with one interval per subtitle block.

Collection Operations
---------------------

An :py:class:`~rustling.srt.SRT` reader behaves like a collection of files.
You can iterate, slice, combine, and modify it:

.. code-block:: python

   import rustling

   srt = rustling.read_srt("path/to/subtitles/")

   # File count and paths
   print(srt.n_files)
   print(srt.file_paths)

   # Iteration and slicing
   for single_file in srt:
       print(single_file.n_files)  # 1

   subset = srt[0:3]

   # Combining
   combined = srt1 + srt2
   srt1 += srt2

   # Appending and extending
   srt1.append(srt2)
   srt1.extend([srt2, srt3])

   # Removing
   last = srt.pop()
   first = srt.pop_left()
   srt.clear()
