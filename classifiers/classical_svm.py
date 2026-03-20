"""
Classical SVM wrapper for QKernel-Benchmark.
"""

import numpy as np
from sklearn.svm import SVC


class ClassicalSVM:
    """Classical SVM wrapper matching the expected interface for the benchmark."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: str | float = "scale", **svc_kwargs):
        self.C = C
        self._clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, **svc_kwargs)

    def fit(self, X_train: np.ndarray, y: np.ndarray) -> "ClassicalSVM":
        """Fit the SVM."""
        self._clf.fit(X_train, y)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self._clf.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        return self._clf.predict_proba(X_test)

    def score(self, X_test: np.ndarray, y: np.ndarray) -> float:
        return self._clf.score(X_test, y)
