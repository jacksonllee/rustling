Language Models
===============

Rustling provides n-gram language models with various smoothing methods.

MLE (Maximum Likelihood Estimation)
------------------------------------

The ``MLE`` model uses raw counts to estimate probabilities, with no smoothing.

.. code-block:: python

   from rustling.lm import MLE

   model = MLE(order=2)
   model.fit([
       ["the", "cat", "sat"],
       ["the", "dog", "ran"],
   ])

   # Score a word given context
   print(model.score("cat", ["the"]))   # 0.5
   print(model.score("dog", ["the"]))   # 0.5

   # Log probability (base 2)
   print(model.logscore("cat", ["the"]))  # -1.0

Lidstone Smoothing
------------------

The ``Lidstone`` model adds a constant ``gamma`` to all counts,
ensuring non-zero probabilities for unseen n-grams.

.. code-block:: python

   from rustling.lm import Lidstone

   model = Lidstone(order=2, gamma=0.1)
   model.fit([
       ["the", "cat", "sat"],
       ["the", "dog", "ran"],
   ])

   # Unseen n-grams get non-zero probability
   print(model.score("bird", ["the"]))  # > 0

Laplace Smoothing
-----------------

The ``Laplace`` model is Lidstone smoothing with ``gamma=1``.

.. code-block:: python

   from rustling.lm import Laplace

   model = Laplace(order=2)
   model.fit([
       ["the", "cat", "sat"],
       ["the", "dog", "ran"],
   ])

   print(model.score("cat", ["the"]))

Text Generation
---------------

All models support text generation via weighted random sampling.

.. code-block:: python

   from rustling.lm import MLE

   model = MLE(order=2)
   model.fit([
       ["the", "cat", "sat", "on", "the", "mat"],
       ["the", "dog", "ran", "to", "the", "park"],
   ])

   # Generate words with a random seed for reproducibility
   words = model.generate(num_words=5, random_seed=42)
   print(words)

   # Generate with a text seed (starting context)
   words = model.generate(num_words=3, text_seed=["the"], random_seed=42)
   print(words)
