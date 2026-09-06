"""
Tests for Error metrics and evaluation functions.
"""

import unittest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jackofalltrades.Errors import (
    mse,
    rmse,
    mae,
    r2score,
    accuracy,
    precision,
    recall,
    f1score,
    mape,
    adjusted_r2score,
    soae,
    soe,
    cross_entropy,
    Error,
)


def _to_python_float(value):
    """Convert JAX Array or numpy array to Python float."""
    if hasattr(value, "item"):  # JAX Array or numpy array
        return float(value.item())
    return float(value)


class TestRegressionMetrics(unittest.TestCase):
    """Tests for regression metrics."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    def test_mse(self):
        """Test Mean Squared Error."""
        result = mse(self.y_true, self.y_pred)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)

    def test_rmse(self):
        """Test Root Mean Squared Error."""
        result = rmse(self.y_true, self.y_pred)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        # RMSE should be sqrt of MSE
        mse_val = mse(self.y_true, self.y_pred)
        mse_float = _to_python_float(mse_val)
        self.assertAlmostEqual(result_float, np.sqrt(mse_float), places=5)

    def test_mae(self):
        """Test Mean Absolute Error."""
        result = mae(self.y_true, self.y_pred)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)

    def test_r2score(self):
        """Test R-squared score."""
        result = r2score(self.y_true, self.y_pred)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        # R2 can be negative for bad models
        self.assertLessEqual(result_float, 1.0)

    def test_adjusted_r2score(self):
        """Test Adjusted R-squared score."""
        n, p = len(self.y_true), 2
        result = adjusted_r2score(self.y_true, self.y_pred, n, p)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)

    def test_mape(self):
        """Test Mean Absolute Percentage Error."""
        # Avoid division by zero
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        result = mape(y_true, y_pred)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)

    def test_soae(self):
        """Test Sum of Absolute Errors."""
        result = soae(self.y_true, self.y_pred)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)

    def test_soe(self):
        """Test Sum of Errors."""
        result = soe(self.y_true, self.y_pred)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)


class TestClassificationMetrics(unittest.TestCase):
    """Tests for classification metrics."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Binary classification
        self.y_true_binary = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        self.y_pred_binary = np.array([0, 1, 0, 1, 1, 1, 1, 0])

        # Multi-class classification
        self.y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        self.y_pred_multi = np.array([0, 1, 2, 0, 1, 1, 0, 1])

    def test_accuracy_binary(self):
        """Test accuracy for binary classification."""
        result = accuracy(self.y_true_binary, self.y_pred_binary)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_accuracy_multi(self):
        """Test accuracy for multi-class classification."""
        result = accuracy(self.y_true_multi, self.y_pred_multi)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_precision_binary(self):
        """Test precision for binary classification."""
        result = precision(self.y_true_binary, self.y_pred_binary)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_precision_multi(self):
        """Test precision for multi-class classification."""
        result = precision(self.y_true_multi, self.y_pred_multi, average="macro")
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_recall_binary(self):
        """Test recall for binary classification."""
        result = recall(self.y_true_binary, self.y_pred_binary)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_recall_multi(self):
        """Test recall for multi-class classification."""
        result = recall(self.y_true_multi, self.y_pred_multi, average="macro")
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_f1score_binary(self):
        """Test F1 score for binary classification."""
        result = f1score(self.y_true_binary, self.y_pred_binary)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_f1score_multi(self):
        """Test F1 score for multi-class classification."""
        result = f1score(self.y_true_multi, self.y_pred_multi, average="macro")
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_pandas_series_input(self):
        """Test that metrics work with pandas Series."""
        y_true_series = pd.Series(self.y_true_binary)
        y_pred_series = pd.Series(self.y_pred_binary)

        result = accuracy(y_true_series, y_pred_series)
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)


class TestErrorClass(unittest.TestCase):
    """Tests for Error class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    def test_initialization(self):
        """Test Error class initialization."""
        error = Error(self.y_true, self.y_pred)
        self.assertIsNotNone(error.y_true)
        self.assertIsNotNone(error.y_predicted)

    def test_mse_method(self):
        """Test MSE method."""
        error = Error(self.y_true, self.y_pred)
        result = error.MSE()
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)

    def test_rmse_method(self):
        """Test RMSE method."""
        error = Error(self.y_true, self.y_pred)
        result = error.RMSE()
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)

    def test_mae_method(self):
        """Test MAE method."""
        error = Error(self.y_true, self.y_pred)
        result = error.MAE()
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)

    def test_rsquared_method(self):
        """Test R-squared method."""
        error = Error(self.y_true, self.y_pred)
        result = error.RSquared()
        result_float = _to_python_float(result)
        self.assertIsInstance(result_float, float)
        self.assertLessEqual(result_float, 1.0)


if __name__ == "__main__":
    unittest.main()
