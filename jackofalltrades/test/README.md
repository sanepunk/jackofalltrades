# Test Suite for jackofalltrades

This directory contains comprehensive tests for all modules in the jackofalltrades package. Tests are designed to be **computationally efficient** with small datasets and minimal training epochs.

## Test Structure

```
test/
├── __init__.py              # Test package initialization
├── test_all.py              # Main test runner for all tests
├── test_regression.py       # Tests for regression models
├── test_classification.py   # Tests for classification models
├── test_errors.py          # Tests for error metrics
├── test_gan.py             # Tests for GAN model
├── test_neighbors.py       # Tests for KNN classifier
├── test_linear_model.py    # Tests for linear model module
└── tests.py                # Legacy tests (backward compatibility)
```

## Running Tests

### Run All Tests

```bash
# From project root
python -m pytest jackofalltrades/test/

# Or using unittest
python -m unittest discover jackofalltrades/test

# Or run the test runner directly
python -m jackofalltrades.test.test_all
```

### Run Specific Test Modules

```bash
# Run regression tests only
python -m unittest jackofalltrades.test.test_regression

# Run classification tests only
python -m unittest jackofalltrades.test.test_classification

# Run error metrics tests only
python -m unittest jackofalltrades.test.test_errors
```

### Run Specific Test Classes

```bash
# Run specific test class
python -m unittest jackofalltrades.test.test_regression.TestLinearRegression

# Run specific test method
python -m unittest jackofalltrades.test.test_regression.TestLinearRegression.test_fit_predict
```

## Test Design Principles

### Computational Efficiency

All tests are designed to be fast and computationally efficient:

1. **Small Datasets**: Tests use 20-40 samples instead of thousands
2. **Minimal Epochs**: Training uses 1-50 epochs instead of thousands
3. **Small Models**: Neural networks use minimal hidden units (4-16)
4. **Small Images**: Image tests use minimal batch sizes (8-16 samples)

### Coverage

Tests cover:

- ✅ All regression models (Linear, Logistic, Ridge, MLP, Adaptive)
- ✅ All classification models (KNN, ImageClassification, DecisionTree)
- ✅ All error metrics (MSE, RMSE, MAE, R², Accuracy, Precision, Recall, F1)
- ✅ GAN model (initialization, training, generation, save/load)
- ✅ Neighbors module (KNN classifier)
- ✅ Linear model module (AdaptiveRegression)
- ✅ Input/output formats (numpy, pandas, torch, jax)

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test model workflows (fit → predict → evaluate)
3. **Edge Cases**: Test with different input formats, single samples, etc.
4. **Save/Load Tests**: Verify model persistence

## Example Test Output

```
$ python -m unittest jackofalltrades.test.test_regression

.....
----------------------------------------------------------------------
Ran 5 tests in 2.345s

OK
```

## Adding New Tests

When adding new features, follow these guidelines:

1. **Use small datasets** (20-50 samples max)
2. **Limit training epochs** (1-50 for neural networks)
3. **Use descriptive test names** (e.g., `test_fit_predict`, `test_save_load`)
4. **Test both success and edge cases**
5. **Verify output shapes and types**
6. **Use assertions for validation**

Example:

```python
def test_new_feature(self):
    """Test description."""
    # Setup
    model = NewModel()
    X = np.random.randn(20, 4)
    y = np.random.randn(20)
    
    # Test
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Assertions
    self.assertEqual(len(predictions), len(y))
    self.assertIsInstance(predictions, np.ndarray)
```

## Continuous Integration

These tests are designed to run quickly in CI/CD pipelines:

- **Total runtime**: < 30 seconds
- **Memory usage**: < 500 MB
- **No external dependencies**: Uses only package dependencies

## Notes

- Tests use fixed random seeds (`np.random.seed(42)`) for reproducibility
- Some tests may have relaxed assertions (e.g., accuracy > 0) to account for randomness
- GAN tests use minimal training (1 epoch) to verify functionality without heavy computation

