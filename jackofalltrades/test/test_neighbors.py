"""
Tests for KNN classifier from neighbors module.
"""

import unittest
import numpy as np
from jackofalltrades.neighbors.neighbors import KNeighborsClassifier
from jackofalltrades.Errors import accuracy


class TestKNeighborsClassifier(unittest.TestCase):
    """Tests for KNeighborsClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Small dataset
        n_samples = 30
        n_features = 4
        self.X_train = np.random.randn(n_samples, n_features)
        self.y_train = np.random.randint(0, 3, n_samples)  # 3 classes
        self.X_test = np.random.randn(10, n_features)

    def test_initialization(self):
        """Test KNN initialization."""
        model = KNeighborsClassifier(k=3)
        self.assertEqual(model.k, 3)

    def test_fit(self):
        """Test fitting the model."""
        model = KNeighborsClassifier(k=3)
        model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(model.X_train)
        self.assertIsNotNone(model.y_train)

    def test_predict(self):
        """Test prediction."""
        model = KNeighborsClassifier(k=3)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all((predictions >= 0) & (predictions < 3)))

    def test_different_k_values(self):
        """Test with different k values."""
        for k in [1, 3, 5]:
            model = KNeighborsClassifier(k=k)
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test[:5])
            self.assertEqual(len(predictions), 5)

    def test_single_sample_prediction(self):
        """Test prediction on single sample."""
        model = KNeighborsClassifier(k=3)
        model.fit(self.X_train, self.y_train)
        prediction = model.predict(self.X_test[0:1])

        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
