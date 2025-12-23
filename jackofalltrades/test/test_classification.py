"""
Tests for Classification models - computationally efficient.
"""
import unittest
import numpy as np
import torch
from jackofalltrades.Models import ImageClassification, DecisionTree, KNNClassifier
from jackofalltrades.Errors import accuracy, f1score


class TestKNNClassifier(unittest.TestCase):
    """Tests for KNNClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Small dataset
        n_samples = 30
        n_features = 4
        self.X_train = np.random.randn(n_samples, n_features)
        self.y_train = np.random.randint(0, 3, n_samples)  # 3 classes
        self.X_test = np.random.randn(10, n_features)
        
    def test_fit_predict(self):
        """Test KNN fitting and prediction."""
        model = KNNClassifier(k=3)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all((predictions >= 0) & (predictions < 3)))
        
    def test_different_k_values(self):
        """Test KNN with different k values."""
        for k in [1, 3, 5]:
            model = KNNClassifier(k=k)
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test[:5])
            self.assertEqual(len(predictions), 5)


class TestImageClassification(unittest.TestCase):
    """Tests for ImageClassification model."""
    
    def setUp(self):
        """Set up test fixtures with small images."""
        np.random.seed(42)
        # Very small dataset for testing
        n_samples = 20
        self.X = np.random.randn(n_samples, 28, 28, 1).astype(np.float32)
        self.y = np.random.randint(0, 5, n_samples)  # 5 classes
        
    def test_initialization(self):
        """Test model initialization."""
        model = ImageClassification(
            input_shape=(28, 28, 1),
            num_classes=5
        )
        self.assertIsNotNone(model.model)
        
    def test_predict_shape(self):
        """Test prediction output shape."""
        model = ImageClassification(
            input_shape=(28, 28, 1),
            num_classes=5
        )
        predictions = model.predict(self.X[:5])
        predictions_np = np.asarray(predictions)
        self.assertEqual(len(predictions_np), 5)
        self.assertTrue(np.all((predictions_np >= 0) & (predictions_np < 5)))
        
    def test_predict_before_fit(self):
        """Test prediction before fitting (should work but may be random)."""
        model = ImageClassification(
            input_shape=(28, 28, 1),
            num_classes=5
        )
        predictions = model.predict(self.X[:5])
        
        self.assertEqual(len(predictions), 5)
        # Convert to numpy array for comparison
        predictions_np = np.asarray(predictions)
        self.assertTrue(np.all((predictions_np >= 0) & (predictions_np < 5)))
        
    def test_fit_predict_minimal(self):
        """Test fitting with minimal epochs."""
        model = ImageClassification(
            input_shape=(28, 28, 1),
            num_classes=5
        )
        # Fit with very few epochs and small batch
        model.fit(self.X, self.y, epochs=2, batch_size=10, verbose=0)
        predictions = model.predict(self.X[:5])
        
        self.assertEqual(len(predictions), 5)
        # Convert to numpy array for comparison
        predictions_np = np.asarray(predictions)
        self.assertTrue(np.all((predictions_np >= 0) & (predictions_np < 5)))
        
    def test_save_load(self):
        """Test model save and load functionality."""
        import os
        import tempfile
        
        model = ImageClassification(
            input_shape=(28, 28, 1),
            num_classes=5
        )
        model.fit(self.X, self.y, epochs=1, batch_size=10, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pickle')
            model.save(model_path)
            
            # Load model
            model2 = ImageClassification(
                input_shape=(28, 28, 1),
                num_classes=5
            )
            model2.load(model_path)
            
            # Compare predictions
            pred1 = model.predict(self.X[:5])
            pred2 = model2.predict(self.X[:5])
            
            np.testing.assert_array_equal(pred1, pred2)


class TestDecisionTree(unittest.TestCase):
    """Tests for DecisionTree classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 40
        n_features = 4
        self.X = np.random.randn(n_samples, n_features)
        self.y = ((self.X[:, 0] + self.X[:, 1]) > 0).astype(int)
        
    def test_fit_predict(self):
        """Test DecisionTree fitting and prediction."""
        model = DecisionTree(max_depth=3)  # Small depth for efficiency
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))


if __name__ == '__main__':
    unittest.main()

