Rustling
========

.. meta::
   :description:
      Rustling: A High-performance Libray for Computational Linguistics
   :keywords: rustling, computational linguistics, natural language processing, nlp, python,
      word segmentation, part-of-speech tagging, language models, ngrams,
      childes, talkbank, chat,
      elan, eaf,
      textgrid, praat,
      conllu, universal dependencies, ud,
      srt,
      averaged perceptron, hidden markov model, longest string matching,

.. image:: https://img.shields.io/pypi/v/rustling.svg
   :target: https://pypi.org/project/rustling/
   :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/rustling.svg
   :target: https://anaconda.org/conda-forge/rustling
   :alt: conda-forge version

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.19140734.svg
   :target: https://doi.org/10.5281/zenodo.19140734
   :alt: Zenodo

.. raw:: html

   <br/><br/>

.. toctree::
   :hidden:

   ngram
   lm
   HMM <hmm>
   wordseg
   POS Tagging <perceptron_pos_tagger>
   chat
   conllu
   elan
   srt
   textgrid
   api

Rustling is a high-performance library for computational linguistics.
It aims to provide flexible and efficient tools to facilitate further research.

Currently implemented features:

* Sequence modeling:

   - :doc:`N-grams <ngram>` and related :doc:`language models <lm>`
   - :doc:`Hidden Markov model <hmm>`
   - :doc:`Word segmentation <wordseg>`
   - :doc:`Averaged perceptron part-of-speech tagging <perceptron_pos_tagger>`

* Handling richly formatted data,
  supporting cross-format conversion as well as both local and remote sources for data ingestion:

   - :doc:`CHAT <chat>` for TalkBank and CHILDES
   - :doc:`ELAN <elan>` for annotated multimedia data
   - :doc:`TextGrid <textgrid>` for Praat annotations
   - :doc:`CoNLL-U <conllu>` for Universal Dependencies
   - :doc:`SRT <srt>` for SubRip subtitles


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

Rustling is highly performant because it is implemented in Rust under the hood.
For benchmarks comparing Rustling against other Python packages with similar functionalities,
please see `benchmarks <https://github.com/jacksonllee/rustling/tree/main/benchmarks>`_.


Source Code
-----------

The source code is available on `GitHub <https://github.com/jacksonllee/rustling>`_.


License
-------

MIT license
