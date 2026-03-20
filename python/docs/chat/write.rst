.. _chat_write:

Writing CHAT Data
=================

To output CHAT data, a :class:`~rustling.chat.CHAT` object can either export data to local files
or write its data to strings.

.. currentmodule:: rustling.chat.CHAT

.. autosummary::

    to_files
    to_strs

These methods are useful for saving CHAT data for re-use or distribution,
especially when your data or :class:`~rustling.chat.CHAT` object is customized in some way,
e.g., by adding or removing data from an existing dataset, or through
an in-memory CHAT data string -- see :ref:`chat_read`.

.. code-block:: python

    from rustling.chat import CHAT
    chat_data = CHAT.from_strs(["*MOT:\they sweetie ."])
    chat_data.to_files("output_dir/")

If your :class:`~rustling.chat.CHAT` object has data organized in multiple CHAT files,
:meth:`~rustling.chat.CHAT.to_files` writes them all to the given directory:

.. code-block:: python

    import rustling
    brown = rustling.read_chat("path/to/your/local/Brown.zip")
    # Brown has data for Adam, Eve, and Sarah.
    # Now we want to save only Eve and Sarah's data somewhere on disk.
    eve_and_sarah = brown.filter(files="Eve|Sarah")
    eve_and_sarah.to_files("your/new/directory")

By default, filenames are derived from the original source file paths
(e.g., ``foo.cha`` stays ``foo.cha``).
If the data was parsed from in-memory strings without explicit IDs,
numbered names are used (``0001.cha``, ``0002.cha``, etc.).
To override, use the ``filenames`` keyword argument of
:meth:`~rustling.chat.CHAT.to_files`.

If you would like the CHAT data as strings in memory for use cases other than
local file export,
:meth:`~rustling.chat.CHAT.to_strs` is available.
