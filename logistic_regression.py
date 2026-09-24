"""Logistic regression trained from scratch with mini-batch gradient descent.

The implementation uses NumPy only for the model and optimization.  It can
train on a CSV file (such as the Pima Indians Diabetes dataset) or on a small,
reproducible synthetic dataset when no file is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The scaler must be fitted before transform().")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class LogisticRegressionGD:
    """Binary logistic regression optimized with mini-batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 1_000,
        batch_size: int = 32,
        l2_strength: float = 0.0,
        random_state: int = 42,
    ) -> None:
        if learning_rate <= 0 or epochs <= 0 or batch_size <= 0:
            raise ValueError("learning_rate, epochs, and batch_size must be positive.")
        if l2_strength < 0:
            raise ValueError("l2_strength must be non-negative.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_strength = l2_strength
        self.random_state = random_state
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Clipping prevents overflow in exp for very large negative values.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _check_fitted(self) -> None:
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

    def _loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probabilities = self._sigmoid(x @ self.weights_ + self.bias_)
        epsilon = 1e-12
        log_loss = -np.mean(
            y * np.log(probabilities + epsilon)
            + (1 - y) * np.log(1 - probabilities + epsilon)
        )
        regularization = self.l2_strength * np.sum(self.weights_**2) / (2 * len(y))
        return float(log_loss + regularization)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x must be 2-D and have the same number of rows as y.")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("y must contain only binary labels 0 and 1.")

        rng = np.random.default_rng(self.random_state)
        self.weights_ = np.zeros(x.shape[1], dtype=float)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):
            indices = rng.permutation(len(x))
            for start in range(0, len(x), self.batch_size):
                batch = indices[start : start + self.batch_size]
                x_batch, y_batch = x[batch], y[batch]
                probabilities = self._sigmoid(x_batch @ self.weights_ + self.bias_)
                errors = probabilities - y_batch

                gradient_w = (x_batch.T @ errors) / len(batch)
                gradient_w += (self.l2_strength / len(x)) * self.weights_
                gradient_b = float(np.mean(errors))
                self.weights_ -= self.learning_rate * gradient_w
                self.bias_ -= self.learning_rate * gradient_b

            self.loss_history_.append(self._loss(x, y))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._sigmoid(np.asarray(x, dtype=float) @ self.weights_ + self.bias_)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1.")
        return (self.predict_proba(x) >= threshold).astype(int)


def stratified_split(
    x: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data while preserving the proportion of each binary class."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []
    for label in (0, 1):
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)
        n_test = max(1, int(round(len(label_indices) * test_size)))
        test_parts.append(label_indices[:n_test])
        train_parts.append(label_indices[n_test:])
    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
    }


def make_demo_data(
    n_samples: int = 1_000, n_features: int = 4, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create a binary dataset without relying on a machine-learning library."""
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(n_samples, n_features))
    true_weights = np.array([1.4, -1.1, 0.8, -0.6])[:n_features]
    logits = x @ true_weights + 0.2
    probabilities = 1 / (1 + np.exp(-logits))
    return x, (rng.random(n_samples) < probabilities).astype(int)


def load_pima_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the headerless Pima CSV: 8 features followed by outcome.

    In this dataset, zero is a missing value for several medical measurements.
    Missing values are replaced with the training-set column median later.
    """
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2 or np.isnan(data).any():
        raise ValueError("Expected a numeric CSV with one label column and no header.")
    return data[:, :-1], data[:, -1].astype(int)


def run(
    x: np.ndarray,
    y: np.ndarray,
    *,
    zero_as_missing: bool = False,
    random_state: int = 42,
) -> dict[str, float]:
    x_train, x_test, y_train, y_test = stratified_split(x, y, random_state=random_state)
    if zero_as_missing:
        missing_columns = [1, 2, 3, 4, 5]
        for column in missing_columns:
            nonzero = x_train[:, column] != 0
            median = np.median(x_train[nonzero, column])
            x_train[~nonzero, column] = median
            x_test[x_test[:, column] == 0, column] = median

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = LogisticRegressionGD(
        learning_rate=0.05,
        epochs=1_000,
        batch_size=32,
        l2_strength=0.1,
        random_state=random_state,
    ).fit(x_train, y_train)
    metrics = classification_metrics(y_test, model.predict(x_test))
    print(f"Final training loss: {model.loss_history_[-1]:.4f}")
    for name, value in metrics.items():
        print(f"{name:>9}: {value:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to a Pima-style CSV (8 feature columns and an outcome column).",
    )
    args = parser.parse_args()
    if args.data:
        x, y = load_pima_csv(args.data)
        run(x, y, zero_as_missing=True)
    else:
        print("No --data supplied; using a reproducible synthetic binary dataset.")
        run(*make_demo_data())


if __name__ == "__main__":
    main()
