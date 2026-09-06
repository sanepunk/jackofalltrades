"""
Tests for Regression models - computationally efficient with small datasets.
"""

import unittest
import numpy as np
import pandas as pd
from jackofalltrades.Models import (
    LinearRegression,
    LogisticRegression,
    MLPRegressor,
    RidgeRegression,
    AdaptiveRegression,
)
from jackofalltrades.Errors import mse, r2score, accuracy


class TestLinearRegression(unittest.TestCase):
    """Tests for LinearRegression model."""

    def setUp(self):
        """Set up test fixtures with small datasets."""
        np.random.seed(42)
        # Small synthetic dataset
        n_samples = 30
        n_features = 3
        self.X = np.random.randn(n_samples, n_features)
        self.y = self.X @ np.array([1.5, -2.0, 0.5]) + 0.1 * np.random.randn(n_samples)

    def test_fit_predict(self):
        """Test model fitting and prediction."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(len(predictions), len(self.y))
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_shape(self):
        """Test prediction output shape."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:5])

        self.assertEqual(len(predictions), 5)

    def test_r2_score(self):
        """Test R2 score calculation."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        score = r2score(self.y, predictions)

        self.assertGreater(score, -1.0)  # R2 can be negative
        self.assertLessEqual(score, 1.0)


class TestLogisticRegression(unittest.TestCase):
    """Tests for LogisticRegression model."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Small binary classification dataset
        n_samples = 40
        n_features = 4
        self.X = np.random.randn(n_samples, n_features)
        # Create separable classes
        self.y = ((self.X[:, 0] + self.X[:, 1]) > 0).astype(int)

    def test_fit_predict(self):
        """Test model fitting and prediction."""
        model = LogisticRegression(max_iter=100)  # Reduced epochs
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_predict_binary(self):
        """Test that predictions are binary."""
        model = LogisticRegression(max_iter=100)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])

        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_accuracy(self):
        """Test accuracy calculation."""
        model = LogisticRegression(max_iter=100)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        acc = accuracy(self.y, predictions)

        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


class TestRidgeRegression(unittest.TestCase):
    """Tests for RidgeRegression model."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 25
        n_features = 3
        self.X = np.random.randn(n_samples, n_features)
        self.y = self.X @ np.array([1.0, -1.5, 0.8]) + 0.05 * np.random.randn(n_samples)

    def test_fit_predict(self):
        """Test Ridge regression fitting and prediction."""
        model = RidgeRegression(epochs=50, learning_rate=0.01)  # Reduced epochs
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(len(predictions), len(self.y))
        self.assertIsInstance(predictions, np.ndarray)


class TestMLPRegressor(unittest.TestCase):
    """Tests for MLPRegressor model."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 20
        n_features = 5
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(n_samples)

    def test_fit_predict(self):
        """Test MLP fitting and prediction with minimal epochs."""
        model = MLPRegressor(
            epochs=10,  # Very small for testing
            hidden_layers=1,
            hidden_units=4,  # Small network
            learning_rate=0.01,
        )
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(len(predictions), len(self.y))
        self.assertIsInstance(predictions, np.ndarray)


class TestAdaptiveRegression(unittest.TestCase):
    """Tests for AdaptiveRegression model."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 30
        n_features = 4
        self.X = np.random.randn(n_samples, n_features)
        self.y = self.X @ np.array([2.0, -1.0, 0.5, 1.5]) + 0.1 * np.random.randn(
            n_samples
        )

    def test_fit_predict(self):
        """Test Adaptive regression with reduced epochs."""
        model = AdaptiveRegression(
            epochs=20,  # Reduced for testing
            learning_rate=0.01,
            early_stop_patience=5,
        )
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(len(predictions), len(self.y))
        self.assertIsInstance(predictions, np.ndarray)


if __name__ == "__main__":
    unittest.main()
