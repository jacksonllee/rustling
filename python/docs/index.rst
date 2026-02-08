Rustling
========

Rustling is a blazingly fast library for computational linguistics.
It is written in Rust, with Python bindings.


Performance
-----------

Benchmarked against pure Python implementataions from NLTK and wordseg.
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
     - **25–39x**
     - NLTK
   * - **Word Segmentation**
     - LongestStringMatching
     - **14x**
     - wordseg
   * -
     - RandomSegmenter
     - **12x**
     - wordseg
   * - **POS Tagging**
     - Training
     - **5x**
     - NLTK
   * -
     - Tagging
     - **6x**
     - NLTK

Installation
------------

.. code-block:: bash

   pip install rustling


Sections
--------

.. toctree::
   :maxdepth: 2

   lm
   wordseg
   tagging
   api


Source Code
-----------

The source code is available on `GitHub <https://github.com/jacksonllee/rustling>`_.
