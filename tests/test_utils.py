"""Unit tests for small helpers."""

import numpy as np
import pytest

from alira.active_learner import _normalize_embeddings


def test_normalize_embeddings_none():
    assert _normalize_embeddings(None) is None


def test_normalize_embeddings_2d_array():
    arr = np.random.randn(5, 8)
    result = _normalize_embeddings(arr)
    assert result is arr
    assert result.shape == (5, 8)


def test_normalize_embeddings_object_array_of_vectors():
    vectors = [np.random.randn(8) for _ in range(5)]
    arr = np.array(vectors, dtype=object)
    result = _normalize_embeddings(arr)
    assert result.shape == (5, 8)
    assert np.allclose(result[0], vectors[0])


def test_normalize_embeddings_empty_object_array():
    arr = np.array([], dtype=object)
    result = _normalize_embeddings(arr)
    assert result.shape == (0,)


def test_normalize_embeddings_series():
    vectors = [np.random.randn(8) for _ in range(3)]
    series = __import__("pandas", fromlist=["Series"]).Series(vectors)
    result = _normalize_embeddings(series)
    assert result.shape == (3, 8)
