CoNLL-U (Universal Dependencies)
=================================

The ``rustling.conllu`` module provides tools for parsing
`CoNLL-U <https://universaldependencies.org/format.html>`_ files,
the standard format for `Universal Dependencies <https://universaldependencies.org/>`_ datasets.

A CoNLL-U file is a plain-text, tab-separated format where sentences are
separated by blank lines. Each token line has 10 fields:
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, and MISC.
Comment lines start with ``#``.

.. code-block:: text

   # sent_id = 1
   # text = The cat sat on the mat.
   1   The     the     DET     DT      Definite=Def|PronType=Art   2   det     _   _
   2   cat     cat     NOUN    NN      Number=Sing                 3   nsubj   _   _
   3   sat     sit     VERB    VBD     Mood=Ind|Tense=Past         0   root    _   _
   4   on      on      ADP     IN      _                           6   case    _   _
   5   the     the     DET     DT      Definite=Def|PronType=Art   6   det     _   _
   6   mat     mat     NOUN    NN      Number=Sing                 3   nmod    _   _
   7   .       .       PUNCT   .       _                           3   punct   _   SpaceAfter=No

Loading Data
------------

:func:`~rustling.read_conllu`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quickest way to load CoNLL-U data is with :func:`~rustling.read_conllu`.
It accepts a file path, directory, ZIP archive, git URL, or HTTP URL
and figures out the right loading strategy automatically:

.. code-block:: python

   import rustling

   # From a local .conllu file
   conllu = rustling.read_conllu("path/to/data.conllu")

   # From a directory (recursively finds all .conllu files)
   conllu = rustling.read_conllu("path/to/ud-treebank/")

   # From a ZIP archive
   conllu = rustling.read_conllu("path/to/treebank.zip")

   # From a git repository (e.g., a Universal Dependencies treebank)
   conllu = rustling.read_conllu("https://github.com/UniversalDependencies/UD_English-EWT.git")

   # From a URL (ZIP files are automatically detected and extracted)
   conllu = rustling.read_conllu("https://example.com/treebank.zip")

Using the class methods directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need finer control -- for example, to pass specific files,
filter by regex, change the file extension, control caching, or parse
in-memory strings -- use the :py:class:`~rustling.conllu.CoNLLU` class methods directly:

.. code-block:: python

   from rustling.conllu import CoNLLU

From specific files:

.. code-block:: python

   conllu = CoNLLU.from_files(["path/to/train.conllu", "path/to/test.conllu"])

From a directory with a regex filter:

.. code-block:: python

   conllu = CoNLLU.from_dir("path/to/treebank/", match=r"test")

The ``extension`` parameter controls which file extension to look for (default: ``".conllu"``).

From a ZIP archive:

.. code-block:: python

   conllu = CoNLLU.from_zip("path/to/treebank.zip")

From a git repository:

.. code-block:: python

   conllu = CoNLLU.from_git("https://github.com/UniversalDependencies/UD_English-EWT.git")

From a URL (ZIP files are automatically detected and extracted):

.. code-block:: python

   conllu = CoNLLU.from_url("https://example.com/treebank.zip")

From in-memory strings:

.. code-block:: python

   conllu = CoNLLU.from_strs([conllu_string_1, conllu_string_2])

Parallel processing
^^^^^^^^^^^^^^^^^^^

All loading methods accept a ``parallel`` parameter (default: ``True``)
to enable parallel parsing of multiple files.

Accessing Data
--------------

Sentences
^^^^^^^^^

Call :py:meth:`~rustling.conllu.CoNLLU.sentences` to get a flat list of all
sentences across all files:

.. code-block:: python

   import rustling

   conllu = rustling.read_conllu("treebank.conllu")

   for sentence in conllu.sentences():
       print(sentence.comments)  # list[str] or None
       for token in sentence.tokens():
           print(token.id, token.form, token.lemma, token.upos, token.deprel)

Tokens
^^^^^^

A :py:class:`~rustling.conllu.Token` has the following properties, corresponding
to the 10 CoNLL-U fields:

- ``id`` -- Word index (integer, range like ``"1-2"`` for multiword tokens, or decimal like ``"1.1"`` for empty nodes).
- ``form`` -- Word form or punctuation symbol.
- ``lemma`` -- Lemma or stem of the word.
- ``upos`` -- Universal POS tag.
- ``xpos`` -- Language-specific POS tag, or ``"_"``.
- ``feats`` -- Morphological features, or ``"_"``.
- ``head`` -- Head of the current word (``"0"`` for root), or ``"_"``.
- ``deprel`` -- Universal dependency relation to HEAD, or ``"_"``.
- ``deps`` -- Enhanced dependency graph, or ``"_"``.
- ``misc`` -- Any other annotation, or ``"_"``.

Comments
^^^^^^^^

A :py:class:`~rustling.conllu.Sentence` has a ``comments`` property that returns
the comment lines (without the leading ``#``), or ``None`` if there are no comments:

.. code-block:: python

   sentence = conllu.sentences()[0]
   if sentence.comments:
       for comment in sentence.comments:
           print(comment)  # e.g., "sent_id = 1" or "text = The cat sat."

Converting to CHAT
------------------

A :py:class:`~rustling.conllu.CoNLLU` reader can convert its data to CHAT format
for use with `CHILDES <https://childes.talkbank.org/>`_ / TalkBank tools.

.. code-block:: python

   import rustling

   conllu = rustling.read_conllu("treebank.conllu")

   # Convert to a CHAT object
   chat = conllu.to_chat()

   # Or get CHAT-formatted strings
   chat_strs = conllu.to_chat_strs()

   # Or write .cha files directly
   conllu.to_chat_files("output_dir/")

The conversion maps CoNLL-U token fields to CHAT morphology and grammar tiers:

- ``%mor`` tier: ``UPOS|LEMMA`` (with ``&FEATS`` appended if features are present)
- ``%gra`` tier: ``ID|HEAD|DEPREL``

Since CoNLL-U files have no participant information, a default participant code
``"SPK"`` (Speaker) is used.

Collection Operations
---------------------

A :py:class:`~rustling.conllu.CoNLLU` reader behaves like a collection of files.
You can iterate, slice, combine, and modify it:

.. code-block:: python

   import rustling

   conllu = rustling.read_conllu("path/to/treebank/")

   # File count and paths
   print(conllu.n_files)
   print(conllu.file_paths)

   # Iteration and slicing
   for single_file in conllu:
       print(single_file.n_files)  # 1

   subset = conllu[0:3]

   # Combining
   combined = conllu1 + conllu2
   conllu1 += conllu2

   # Appending and extending
   conllu1.append(conllu2)
   conllu1.extend([conllu2, conllu3])

   # Removing
   last = conllu.pop()
   first = conllu.pop_left()
   conllu.clear()
