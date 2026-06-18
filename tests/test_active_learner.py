"""Smoke tests for the ActiveLearner class."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alira import ActiveLearner


@pytest.fixture
def corpus():
    return ["sports news", "superhero movie", "finance report", "action film", "batman returns"]


@pytest.fixture
def embeddings(corpus):
    return np.random.randn(len(corpus), 8)


@pytest.fixture
def mocks():
    def fake_embeddings(texts):
        return np.random.randn(len(texts), 8).tolist()

    def fake_evaluate(query, texts, prompt=None):
        return ["hero" in t.lower() or "batman" in t.lower() or "action" in t.lower() for t in texts]

    def fake_synthetic(query, n, examples, prompt=None, max_retries=3):
        return [f"synthetic {i} about {query}" for i in range(n)]

    with (
        patch("alira.active_learner.send_embedding_request", side_effect=fake_embeddings) as emb,
        patch("alira.active_learner.evaluate", side_effect=fake_evaluate) as ev,
        patch("alira.active_learner.generate_synthetic_texts", side_effect=fake_synthetic) as syn,
    ):
        yield {"embeddings": emb, "evaluate": ev, "synthetic": syn}


def test_active_learner_fit_and_predict(corpus, embeddings, mocks):
    learner = ActiveLearner(
        corpus=corpus,
        embeddings=embeddings,
        min_iterations=1,
        max_iterations=2,
        n_eval_per_iteration=2,
    )
    result = learner.fit(query="superheroes")

    assert result is learner
    assert learner.query_ == "superheroes"
    assert learner.classifier_ is not None
    assert learner.iterations_ >= 1

    scores = learner.predict_proba()
    assert isinstance(scores, pd.Series)
    assert len(scores) == len(corpus)
    assert scores.between(0, 1).all()

    predictions = learner.predict()
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(corpus)
    assert set(predictions.unique()).issubset({True, False})


def test_active_learner_unfitted_predict_raises(corpus):
    learner = ActiveLearner(corpus=corpus)
    with pytest.raises(ValueError, match="not fitted yet"):
        learner.predict_proba()


def test_active_learner_rejects_both_corpus_and_embeddings(corpus, embeddings, mocks):
    learner = ActiveLearner(corpus=corpus, embeddings=embeddings, min_iterations=1, max_iterations=1, n_eval_per_iteration=1)
    learner.fit(query="superheroes")

    new_corpus = ["new text one", "new text two"]
    new_embeddings = np.random.randn(len(new_corpus), 8)

    with pytest.raises(ValueError, match="Cannot specify both"):
        learner.predict_proba(corpus=new_corpus, embeddings=new_embeddings)


def test_active_learner_predict_proba_on_new_corpus(corpus, embeddings, mocks):
    learner = ActiveLearner(corpus=corpus, embeddings=embeddings, min_iterations=1, max_iterations=1, n_eval_per_iteration=1)
    learner.fit(query="superheroes")

    new_corpus = ["new text one", "new text two"]
    scores = learner.predict_proba(corpus=new_corpus)
    assert len(scores) == len(new_corpus)
    assert scores.between(0, 1).all()
