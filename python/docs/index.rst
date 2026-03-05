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
It is written in Rust, with Python bindings.

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

.. code-block:: bash

   pip install rustling


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
     - **10x**
     - NLTK
   * - 
     - Score
     - **1.9x**
     - NLTK
   * - 
     - Generate
     - **106--114x**
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
     - **18x**
     - NLTK
   * - **HMM**
     - Fit
     - **13x**
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
     - **43x**
     - pylangacq
   * -
     - Reading from strings
     - **70x**
     - pylangacq
   * -
     - Parsing utterances
     - **15x**
     - pylangacq
   * -
     - Parsing tokens
     - **9x**
     - pylangacq


Source Code
-----------

The source code is available on `GitHub <https://github.com/jacksonllee/rustling>`_.
