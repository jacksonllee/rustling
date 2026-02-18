N-grams
=======

Rustling provides an efficient n-gram counter for extracting and counting n-gram frequencies
from sequences.

Basic Usage
-----------

The :py:class:`~rustling.ngram.Ngrams` class counts n-grams from sequences of strings.

.. code-block:: python

   from rustling.ngram import Ngrams

   ng = Ngrams(n=2)
   ng.count(["the", "cat", "sat"])
   ng.count(["the", "dog", "ran"])

   print(ng[("the", "cat")])  # 1
   print(ng[("the", "dog")])  # 1

   # Most common bigrams
   print(ng.most_common(2))
   # [(('the', 'cat'), 1), (('the', 'dog'), 1)]

Counting from Multiple Sequences
---------------------------------

Use :py:meth:`~rustling.ngram.Ngrams.count_seqs` to count n-grams from multiple sequences at once.

.. code-block:: python

   from rustling.ngram import Ngrams

   ng = Ngrams(n=2)
   ng.count_seqs([
       ["the", "cat", "sat"],
       ["the", "dog", "ran"],
       ["the", "cat", "ran"],
   ])

   print(ng[("the", "cat")])  # 2
   print(ng.total())          # 6

Mixed Orders
------------

Set ``min_n`` to collect n-grams of multiple orders simultaneously.

.. code-block:: python

   from rustling.ngram import Ngrams

   ng = Ngrams(n=3, min_n=1)
   ng.count(["a", "b", "c"])

   # Unigrams, bigrams, and trigrams are all counted
   print(ng.most_common(order=1))  # unigrams
   print(ng.most_common(order=2))  # bigrams
   print(ng.most_common(order=3))  # trigrams

Converting to Counter
---------------------

Use :py:meth:`~rustling.ngram.Ngrams.to_counter` to get a standard ``collections.Counter``.

.. code-block:: python

   from rustling.ngram import Ngrams

   ng = Ngrams(n=2)
   ng.count_seqs([
       ["the", "cat", "sat"],
       ["the", "dog", "ran"],
   ])

   counter = ng.to_counter()
   print(counter)
   # Counter({('the', 'cat'): 1, ('cat', 'sat'): 1, ('the', 'dog'): 1, ('dog', 'ran'): 1})

Combining Counters
------------------

``Ngrams`` objects can be combined with ``+`` or ``+=``.

.. code-block:: python

   from rustling.ngram import Ngrams

   ng1 = Ngrams(n=2)
   ng1.count(["the", "cat", "sat"])

   ng2 = Ngrams(n=2)
   ng2.count(["the", "dog", "ran"])

   combined = ng1 + ng2
   print(combined.total())  # 4
