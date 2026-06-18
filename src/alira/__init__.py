"""ALIRA – Active Learning Iterative Retrieval Agent package.

Tools for building datasets with embeddings and running active learning
classification on documents. ALIRA uses active learning to iteratively
discover relevant documents from large corpora using LLM validation and
classifier refinement.
"""

from alira.active_learner import ActiveLearner
