Hidden Markov Model
===================

The :py:class:`~rustling.hmm.HiddenMarkovModel` supports both unsupervised
training (Baum-Welch EM algorithm) and supervised training (labeled counting
with Laplace smoothing). It uses the Viterbi algorithm for decoding and the
Forward algorithm for computing log-likelihoods.

.. code-block:: python

   from rustling.hmm import HiddenMarkovModel

   # Initialize with the number of hidden states
   model = HiddenMarkovModel(n_states=3, random_seed=42)

   # Fit on unlabeled observation sequences
   model.fit([
       ["The", "cat", "sat"],
       ["A", "dog", "ran"],
       ["The", "dog", "sat"],
   ])

   # Predict hidden state sequences (batch)
   result = model.predict([["The", "dog", "sat"]])
   print(result)
   # [[0, 2, 1]]

   # Score sequences (log-likelihood, batch)
   scores = model.score([["The", "dog", "sat"]])
   print(scores)
   # [-5.545...]
