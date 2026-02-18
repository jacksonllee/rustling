Rustling
========

Rustling is a blazingly fast library for computational linguistics.
It is written in Rust, with Python bindings.

.. toctree::
   :maxdepth: 1

   lm
   ngram
   wordseg
   tagging
   chat
   api


Installation
------------

.. code-block:: bash

   pip install rustling


Performance
-----------

Benchmarked against pure Python implementations from NLTK, wordseg (v0.0.5), and pylangacq (v0.19.1).
See `benchmarks/ <https://github.com/jacksonllee/rustling/tree/main/benchmarks>`_
for full details and reproduction scripts.

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Component
     - Task
     - Speedup
     - vs.
   * - **Language Models**
     - Fit
     - **10x**
     - NLTK
   * -
     - Score
     - **2x**
     - NLTK
   * -
     - Generate
     - **80–112x**
     - NLTK
   * - **Word Segmentation**
     - LongestStringMatching
     - **9x**
     - wordseg
   * -
     - RandomSegmenter
     - **1.1x**
     - wordseg
   * - **POS Tagging**
     - Training
     - **5x**
     - NLTK
   * -
     - Tagging
     - **7x**
     - NLTK
   * - **CHAT Parsing**
     - from_dir
     - **55x**
     - pylangacq
   * -
     - from_zip
     - **48x**
     - pylangacq
   * -
     - from_files
     - **63x**
     - pylangacq
   * -
     - from_strs
     - **116x**
     - pylangacq
   * -
     - words()
     - **3x**
     - pylangacq
   * -
     - utterances()
     - **15x**
     - pylangacq


Source Code
-----------

The source code is available on `GitHub <https://github.com/jacksonllee/rustling>`_.
