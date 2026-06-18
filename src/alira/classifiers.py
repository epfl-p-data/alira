from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


class AbstractClassifier(ABC):
    """Abstract base class for binary classifiers."""

    @abstractmethod
    def fit(self, X: NDArray[np.float64], y: NDArray[np.bool_]) -> None:
        """Train the classifier on labeled data."""

    @abstractmethod
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probability of the positive class."""


class LogisticRegressionClassifier(AbstractClassifier):
    """Logistic regression classifier with balanced class weights."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = c
        self.model = LogisticRegression(
            C=c,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )

    def fit(self, X: NDArray[np.float64], y: NDArray[np.bool_]) -> None:
        """Train the logistic regression model."""
        self.model.fit(X, y)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
