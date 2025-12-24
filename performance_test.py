"""
Performance Testing Script for jackofalltrades Models
Compares model fitting time between custom implementations and sklearn equivalents.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

# Custom implementations
from jackofalltrades.Models import (
    KNNClassifier, MLPRegressor, LinearRegression, 
    RidgeRegression, LogisticRegression
)
from jackofalltrades.linear_model import AdaptiveRegression

# Sklearn equivalents
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
from sklearn.linear_model import (
    LinearRegression as SklearnLinearRegression,
    Ridge as SklearnRidge,
    LogisticRegression as SklearnLogisticRegression
)
# from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree


def time_fit(model, X, y, model_name):
    """Measure the time taken to fit a model."""
    start_time = time.perf_counter()
    model.fit(X, y)
    end_time = time.perf_counter()
    return end_time - start_time


def generate_regression_data(n_samples=10000, n_features=10, noise=0.1):
    """Generate synthetic regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=42
    )
    return X, y


def generate_classification_data(n_samples=10000, n_features=10, n_classes=2):
    """Generate synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features,
        n_redundant=0,
        random_state=42
    )
    return X, y


def test_knn_classifier():
    """Test KNN Classifier performance."""
    print("\n" + "="*70)
    print("KNN Classifier Performance Test")
    print("="*70)
    
    X, y = generate_classification_data(n_samples=10000, n_features=10)
    
    # Custom implementation
    custom_model = KNNClassifier(k=5)
    custom_model.fit(X, y)
    time_counter = time.perf_counter()
    pred = custom_model.predict(X)
    custom_time = time.perf_counter() - time_counter
    
    # Sklearn implementation
    sklearn_model = KNeighborsClassifier(n_neighbors=5)
    sklearn_model.fit(X, y)
    time_counter = time.perf_counter()
    pred = sklearn_model.predict(X)
    sklearn_time = time.perf_counter() - time_counter
    
    print(f"Custom KNN fit time:  {custom_time:.6f} seconds")
    print(f"Sklearn KNN fit time: {sklearn_time:.6f} seconds")
    print(f"Speedup ratio:        {sklearn_time/custom_time:.2f}x")
    print(f"Winner: {'Custom' if custom_time < sklearn_time else 'Sklearn'}")


def test_mlp_regressor():
    """Test MLP Regressor performance."""
    print("\n" + "="*70)
    print("MLP Regressor Performance Test")
    print("="*70)
    
    X, y = generate_regression_data(n_samples=10000, n_features=10)
    
    # Custom implementation - using fewer epochs for faster testing
    custom_model = MLPRegressor(
        learning_rate=1e-4,
        epochs=100,  # Reduced for faster testing
        hidden_layers=1,
        hidden_units=10,
        data_regularization=True
    )
    custom_time = time_fit(custom_model, X, y, "Custom MLP")
    
    # Sklearn implementation - matching parameters
    sklearn_model = SklearnMLPRegressor(
        hidden_layer_sizes=(10,),
        max_iter=100,  # Matching epochs
        learning_rate_init=1e-4,
        random_state=42,
        early_stopping=False
    )
    sklearn_time = time_fit(sklearn_model, X, y, "Sklearn MLP")
    
    print(f"Custom MLP fit time:  {custom_time:.6f} seconds")
    print(f"Sklearn MLP fit time: {sklearn_time:.6f} seconds")
    print(f"Speedup ratio:        {sklearn_time/custom_time:.2f}x")
    print(f"Winner: {'Custom' if custom_time < sklearn_time else 'Sklearn'}")


def test_linear_regression():
    """Test Linear Regression performance."""
    print("\n" + "="*70)
    print("Linear Regression Performance Test")
    print("="*70)
    
    X, y = generate_regression_data(n_samples=10000, n_features=10)
    
    # Custom implementation
    custom_model = LinearRegression(data_regularization=True)
    custom_time = time_fit(custom_model, X, y, "Custom Linear")
    
    # Sklearn implementation
    sklearn_model = SklearnLinearRegression()
    sklearn_time = time_fit(sklearn_model, X, y, "Sklearn Linear")
    
    print(f"Custom Linear fit time:  {custom_time:.6f} seconds")
    print(f"Sklearn Linear fit time: {sklearn_time:.6f} seconds")
    print(f"Speedup ratio:           {sklearn_time/custom_time:.2f}x")
    print(f"Winner: {'Custom' if custom_time < sklearn_time else 'Sklearn'}")


def test_ridge_regression():
    """Test Ridge Regression performance."""
    print("\n" + "="*70)
    print("Ridge Regression Performance Test")
    print("="*70)
    
    X, y = generate_regression_data(n_samples=10000, n_features=10)
    
    # Custom implementation
    custom_model = RidgeRegression(
        regularization_strength=1.0,
        data_regularization=True
    )
    custom_time = time_fit(custom_model, X, y, "Custom Ridge")
    
    # Sklearn implementation
    sklearn_model = SklearnRidge(alpha=1.0)
    sklearn_time = time_fit(sklearn_model, X, y, "Sklearn Ridge")
    
    print(f"Custom Ridge fit time:  {custom_time:.6f} seconds")
    print(f"Sklearn Ridge fit time: {sklearn_time:.6f} seconds")
    print(f"Speedup ratio:          {sklearn_time/custom_time:.2f}x")
    print(f"Winner: {'Custom' if custom_time < sklearn_time else 'Sklearn'}")


def test_logistic_regression():
    """Test Logistic Regression performance."""
    print("\n" + "="*70)
    print("Logistic Regression Performance Test")
    print("="*70)
    
    X, y = generate_classification_data(n_samples=10000, n_features=10)
    
    # Custom implementation - using fewer epochs for faster testing
    custom_model = LogisticRegression(
        learning_rate=0.01,
        epochs=500,  # Reduced for faster testing
        data_regularization=True
    )
    custom_time = time_fit(custom_model, X, y, "Custom Logistic")
    
    # Sklearn implementation
    sklearn_model = SklearnLogisticRegression(
        max_iter=500,  # Matching epochs
        random_state=42
    )
    sklearn_time = time_fit(sklearn_model, X, y, "Sklearn Logistic")
    
    print(f"Custom Logistic fit time:  {custom_time:.6f} seconds")
    print(f"Sklearn Logistic fit time: {sklearn_time:.6f} seconds")
    print(f"Speedup ratio:             {sklearn_time/custom_time:.2f}x")
    print(f"Winner: {'Custom' if custom_time < sklearn_time else 'Sklearn'}")


# def test_decision_tree():
#     """Test Decision Tree performance."""
#     print("\n" + "="*70)
#     print("Decision Tree Performance Test")
#     print("="*70)
    
#     X, y = generate_classification_data(n_samples=10000, n_features=10)
    
#     # Custom implementation
#     custom_model = DecisionTree(max_depth=10)
#     custom_time = time_fit(custom_model, X, y, "Custom DecisionTree")
    
#     # Sklearn implementation
#     sklearn_model = SklearnDecisionTree(max_depth=10, random_state=42)
#     sklearn_time = time_fit(sklearn_model, X, y, "Sklearn DecisionTree")
    
#     print(f"Custom DecisionTree fit time:  {custom_time:.6f} seconds")
#     print(f"Sklearn DecisionTree fit time: {sklearn_time:.6f} seconds")
#     print(f"Speedup ratio:                 {sklearn_time/custom_time:.2f}x")
#     print(f"Winner: {'Custom' if custom_time < sklearn_time else 'Sklearn'}")


def test_adaptive_regression():
    """Test Adaptive Regression performance (no sklearn equivalent, just timing)."""
    print("\n" + "="*70)
    print("Adaptive Regression Performance Test")
    print("="*70)
    print("Note: No direct sklearn equivalent - showing custom implementation timing only")
    
    X, y = generate_regression_data(n_samples=10000, n_features=10)
    
    # Custom implementation - using fewer epochs for faster testing
    custom_model = AdaptiveRegression(
        learning_rate=0.01,
        epochs=500,  # Reduced for faster testing
        data_regularization=True
    )
    custom_time = time_fit(custom_model, X, y, "Custom Adaptive")
    
    print(f"Custom Adaptive fit time: {custom_time:.6f} seconds")


def run_all_tests():
    """Run all performance tests."""
    print("\n" + "="*70)
    print("JACKOFALLTRADES PERFORMANCE BENCHMARK")
    print("Testing Model Fitting Time Only")
    print("="*70)
    
    try:
        test_knn_classifier()
    except Exception as e:
        print(f"Error in KNN test: {e}")
    
    try:
        test_mlp_regressor()
    except Exception as e:
        print(f"Error in MLP Regressor test: {e}")
    
    try:
        test_linear_regression()
    except Exception as e:
        print(f"Error in Linear Regression test: {e}")
    
    try:
        test_ridge_regression()
    except Exception as e:
        print(f"Error in Ridge Regression test: {e}")
    
    try:
        test_logistic_regression()
    except Exception as e:
        print(f"Error in Logistic Regression test: {e}")
    
    # try:
    #     test_decision_tree()
    # except Exception as e:
    #     print(f"Error in Decision Tree test: {e}")
    
    try:
        test_adaptive_regression()
    except Exception as e:
        print(f"Error in Adaptive Regression test: {e}")
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()

