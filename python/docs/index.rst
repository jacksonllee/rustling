Rustling
========

.. meta::
   :description:
      Rustling: A Blazingly Fast Libray for Computational Linguistics
   :keywords: rustling, computational linguistics, linguistics,
      natural language processing, nlp, text processing,
      word segmentation, part-of-speech tagging, language models,
      ngrams, childes, talkbank, chat,
      averaged perceptron, hidden markov model, longest string matching,
      rust, python

Rustling is a blazingly fast library for computational linguistics.

.. toctree::
   :maxdepth: 1

   N-grams <ngram>
   Language Models <lm>
   HMM <hmm>
   Word Segmentation <wordseg>
   POS Tagging <perceptron_pos_tagger>
   CHAT Parsing <chat>
   API Reference <api>


Installation
------------

Using pip:

.. code-block:: bash

   pip install rustling

Using conda:

.. code-block:: bash

   conda install -c conda-forge rustling

For Pyodide, pre-built WASM wheels (with multithreading disabled, as Pyodide does not support it)
are available from each `GitHub release <https://github.com/jacksonllee/rustling/releases>`_
— look for the ``.whl`` file with ``emscripten`` in the filename.

Rustling is also available in `Rust <https://docs.rs/rustling>`_.


Performance
-----------

Benchmarked against Python implementations from NLTK, wordseg (v0.0.5),
pylangacq (v0.19.1), and hmmlearn (v0.3.3).
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
     - **11x**
     - NLTK
   * - 
     - Score
     - **2x**
     - NLTK
   * - 
     - Generate
     - **86--107x**
     - NLTK
   * - **Word Segmentation**
     - LongestStringMatching
     - **9x**
     - wordseg
   * - **POS Tagging**
     - Training
     - **5x**
     - NLTK
   * - 
     - Tagging
     - **17x**
     - NLTK
   * - **HMM**
     - Fit
     - **14x**
     - hmmlearn
   * - 
     - Predict
     - **0.9x**
     - hmmlearn
   * - 
     - Score
     - **5x**
     - hmmlearn
   * - **CHAT Parsing**
     - Reading from a ZIP archive
     - **30x**
     - pylangacq
   * - 
     - Reading from strings
     - **35x**
     - pylangacq
   * - 
     - Parsing utterances
     - **15x**
     - pylangacq
   * - 
     - Parsing tokens
     - **8x**
     - pylangacq


Source Code
-----------

The source code is available on `GitHub <https://github.com/jacksonllee/rustling>`_.
