"""
Tests for linear_model module.
"""
import unittest
import numpy as np
from jackofalltrades.linear_model import AdaptiveRegression
from jackofalltrades.Errors import r2score, mse


class TestAdaptiveRegression(unittest.TestCase):
    """Tests for AdaptiveRegression from linear_model."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 30
        n_features = 4
        self.X = np.random.randn(n_samples, n_features)
        self.y = self.X @ np.array([2.0, -1.0, 0.5, 1.5]) + 0.1 * np.random.randn(n_samples)
        
    def test_initialization(self):
        """Test model initialization."""
        model = AdaptiveRegression(
            epochs=20,
            learning_rate=0.01
        )
        self.assertIsNotNone(model)
        
    def test_fit_predict(self):
        """Test fitting and prediction."""
        model = AdaptiveRegression(
            epochs=20,  # Reduced for testing
            learning_rate=0.01,
            early_stop_patience=5
        )
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_with_regularization(self):
        """Test with data regularization."""
        model = AdaptiveRegression(
            epochs=20,
            learning_rate=0.01,
            data_regularization=True
        )
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
        
    def test_without_regularization(self):
        """Test without data regularization."""
        model = AdaptiveRegression(
            epochs=20,
            learning_rate=0.01,
            data_regularization=False
        )
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))


if __name__ == '__main__':
    unittest.main()

