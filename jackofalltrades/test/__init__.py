"""
Test package for jackofalltrades.

This package contains comprehensive tests for all modules.
Tests are designed to be computationally efficient with small datasets.
"""

from .test_regression import (
    TestLinearRegression, TestLogisticRegression, TestRidgeRegression,
    TestMLPRegressor, TestAdaptiveRegression
)
from .test_classification import (
    TestKNNClassifier, TestImageClassification, TestDecisionTree
)
from .test_errors import (
    TestRegressionMetrics, TestClassificationMetrics, TestErrorClass
)
from .test_gan import TestGAN
from .test_neighbors import TestKNeighborsClassifier
from .test_linear_model import TestAdaptiveRegression

__all__ = [
    'TestLinearRegression',
    'TestLogisticRegression',
    'TestRidgeRegression',
    'TestMLPRegressor',
    'TestAdaptiveRegression',
    'TestKNNClassifier',
    'TestImageClassification',
    'TestDecisionTree',
    'TestRegressionMetrics',
    'TestClassificationMetrics',
    'TestErrorClass',
    'TestGAN',
    'TestKNeighborsClassifier',
]
