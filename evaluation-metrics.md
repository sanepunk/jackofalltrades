# 📊 Evaluation Metrics Guide - jackofalltrades

This comprehensive guide covers all evaluation metrics and error analysis tools available in the `jackofalltrades.Errors` module. Understanding how to properly evaluate your machine learning models is crucial for building reliable and effective solutions.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Function-Based Metrics](#function-based-metrics)
3. [Error Class](#error-class)
4. [Regression Metrics](#regression-metrics)
5. [Classification Metrics](#classification-metrics)
6. [Advanced Evaluation](#advanced-evaluation)
7. [Best Practices](#best-practices)

---

## 🚀 Quick Start

The `jackofalltrades.Errors` module provides two ways to compute metrics:

1. **Function-based**: Quick, one-line metric calculations
2. **Class-based**: Comprehensive error analysis with multiple metrics

```python
from jackofalltrades.Errors import accuracy, f1score, Error

# Function-based (classification)
acc = accuracy([1, 0, 1, 1], [1, 0, 0, 1])
f1 = f1score([1, 0, 1, 1], [1, 0, 0, 1])

# Class-based (regression)
evaluator = Error(y_true=[1, 2, 3, 4], y_predicted=[1.1, 1.9, 3.2, 3.8])
mse = evaluator.MSE()
r2 = evaluator.RSquared()
```

---

## 🔢 Function-Based Metrics

Quick and easy metric calculations for common use cases.

### Classification Functions

#### Accuracy Score

**Description**: Percentage of correct predictions.

**Formula**: `(Correct Predictions) / (Total Predictions)`

**Range**: 0 to 1 (higher is better)

```python
from jackofalltrades.Errors import accuracy
import numpy as np

# Binary classification example
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

acc_score = accuracy(y_true, y_pred)
print(f"Accuracy: {acc_score:.3f}")  # 0.750

# Multi-class classification example
y_true_multi = [0, 1, 2, 2, 1, 0, 1, 2]
y_pred_multi = [0, 1, 2, 1, 1, 0, 2, 2]

acc_multi = accuracy(y_true_multi, y_pred_multi)
print(f"Multi-class Accuracy: {acc_multi:.3f}")

# Practical example with model predictions
from jackofalltrades.datasets import get_real_estate
from jackofalltrades.Models import LogisticRegression
from sklearn.model_selection import train_test_split

# Create binary classification problem
X, y = get_real_estate()
y_binary = (y > y.median()).astype(int)  # Convert to binary: expensive vs. cheap

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
final_accuracy = accuracy(y_test, predictions)
print(f"Model Accuracy: {final_accuracy:.3f}")
```

#### F1 Score

**Description**: Harmonic mean of precision and recall.

**Formula**: `2 * (Precision * Recall) / (Precision + Recall)`

**Range**: 0 to 1 (higher is better)

**Best For**: Imbalanced datasets where you care about both precision and recall

```python
from jackofalltrades.Errors import f1score
import numpy as np

# Binary classification example
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

f1 = f1score(y_true, y_pred)
print(f"F1 Score: {f1:.3f}")

# Imbalanced dataset example
np.random.seed(42)
n_samples = 1000

# Create imbalanced dataset (10% positive class)
y_true_imbalanced = np.concatenate([
    np.ones(100),      # 100 positive samples
    np.zeros(900)      # 900 negative samples
])

# Simulate model predictions with some errors
y_pred_imbalanced = y_true_imbalanced.copy()
# Add some false positives
false_positive_idx = np.random.choice(np.where(y_true_imbalanced == 0)[0], 50, replace=False)
y_pred_imbalanced[false_positive_idx] = 1
# Add some false negatives  
false_negative_idx = np.random.choice(np.where(y_true_imbalanced == 1)[0], 20, replace=False)
y_pred_imbalanced[false_negative_idx] = 0

# Compare accuracy vs F1 score
acc_imbalanced = accuracy(y_true_imbalanced, y_pred_imbalanced)
f1_imbalanced = f1score(y_true_imbalanced, y_pred_imbalanced)

print(f"\nImbalanced Dataset Results:")
print(f"Accuracy: {acc_imbalanced:.3f}")
print(f"F1 Score: {f1_imbalanced:.3f}")
print(f"Class distribution: {np.bincount(y_true_imbalanced)}")
```

---

## 📈 Error Class

The `Error` class provides comprehensive error analysis for regression problems.

### Basic Usage

```python
from jackofalltrades.Errors import Error
import numpy as np

# Create sample regression data
y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]

# Initialize Error object
evaluator = Error(y_true=y_true, y_predicted=y_pred)

# Compute all available metrics
print("Regression Metrics:")
print(f"MSE (Mean Squared Error): {evaluator.MSE():.4f}")
print(f"RMSE (Root Mean Squared Error): {evaluator.RMSE():.4f}")
print(f"MAE (Mean Absolute Error): {evaluator.MAE():.4f}")
print(f"R² (R-squared): {evaluator.RSquared():.4f}")
```

---

## 📊 Regression Metrics

### Mean Squared Error (MSE)

**Description**: Average of squared differences between actual and predicted values.

**Formula**: `MSE = (1/n) * Σ(y_true - y_pred)²`

**Range**: 0 to ∞ (lower is better)

**Best For**: Penalizing large errors more than small ones

```python
from jackofalltrades.Errors import Error
from jackofalltrades.datasets import get_real_estate
from jackofalltrades.Models import LinearRegression, RidgeRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load real estate data
X, y = get_real_estate()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train different models
linear_model = LinearRegression()
ridge_model = RidgeRegression()

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Make predictions
linear_pred = linear_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)

# Compare MSE
linear_evaluator = Error(y_true=y_test, y_predicted=linear_pred)
ridge_evaluator = Error(y_true=y_test, y_predicted=ridge_pred)

print("MSE Comparison:")
print(f"Linear Regression MSE: {linear_evaluator.MSE():.2f}")
print(f"Ridge Regression MSE: {ridge_evaluator.MSE():.2f}")

# Understanding MSE sensitivity to outliers
print("\nMSE Sensitivity to Outliers:")

# Normal predictions
normal_pred = [1, 2, 3, 4, 5]
normal_true = [1.1, 2.1, 3.1, 4.1, 5.1]

# Predictions with one outlier
outlier_pred = [1, 2, 3, 4, 10]  # Large error in last prediction
outlier_true = [1.1, 2.1, 3.1, 4.1, 5.1]

normal_mse = Error(y_true=normal_true, y_predicted=normal_pred).MSE()
outlier_mse = Error(y_true=outlier_true, y_predicted=outlier_pred).MSE()

print(f"Normal predictions MSE: {normal_mse:.3f}")
print(f"With outlier MSE: {outlier_mse:.3f}")
print(f"MSE increased by: {outlier_mse / normal_mse:.1f}x")
```

### Root Mean Squared Error (RMSE)

**Description**: Square root of MSE, in the same units as the target variable.

**Formula**: `RMSE = √MSE`

**Range**: 0 to ∞ (lower is better)

**Best For**: Interpretable error metric in original units

```python
from jackofalltrades.Errors import Error
import matplotlib.pyplot as plt

# Example with interpretable units (house prices)
actual_prices = [300000, 450000, 600000, 750000, 900000]  # Dollars
predicted_prices = [320000, 430000, 580000, 780000, 920000]  # Dollars

evaluator = Error(y_true=actual_prices, y_predicted=predicted_prices)

mse = evaluator.MSE()
rmse = evaluator.RMSE()

print(f"House Price Prediction Errors:")
print(f"MSE: ${mse:,.0f}² (squared dollars)")
print(f"RMSE: ${rmse:,.0f} (dollars)")
print(f"Average prediction error: ${rmse:,.0f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(actual_prices, predicted_prices, s=100, alpha=0.7)
plt.plot([min(actual_prices), max(actual_prices)], 
         [min(actual_prices), max(actual_prices)], 'r--', linewidth=2)

plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'House Price Predictions (RMSE: ${rmse:,.0f})')
plt.grid(True, alpha=0.3)
plt.show()

# RMSE interpretation
print(f"\nRMSE Interpretation:")
print(f"On average, predictions are off by ${rmse:,.0f}")
print(f"As a percentage of mean price: {rmse / np.mean(actual_prices) * 100:.1f}%")
```

### Mean Absolute Error (MAE)

**Description**: Average of absolute differences between actual and predicted values.

**Formula**: `MAE = (1/n) * Σ|y_true - y_pred|`

**Range**: 0 to ∞ (lower is better)

**Best For**: Robust to outliers, easy to interpret

```python
from jackofalltrades.Errors import Error
import numpy as np

# Compare MAE vs RMSE sensitivity to outliers
print("MAE vs RMSE: Outlier Sensitivity Comparison\n")

# Dataset 1: Normal errors
normal_true = [10, 20, 30, 40, 50]
normal_pred = [12, 18, 32, 38, 52]

# Dataset 2: Same errors + one large outlier
outlier_true = [10, 20, 30, 40, 50]
outlier_pred = [12, 18, 32, 38, 80]  # Large error in last prediction

# Calculate metrics
normal_eval = Error(y_true=normal_true, y_predicted=normal_pred)
outlier_eval = Error(y_true=outlier_true, y_predicted=outlier_pred)

print("Normal Dataset:")
print(f"  MAE: {normal_eval.MAE():.2f}")
print(f"  RMSE: {normal_eval.RMSE():.2f}")

print("\nWith Outlier:")
print(f"  MAE: {outlier_eval.MAE():.2f}")
print(f"  RMSE: {outlier_eval.RMSE():.2f}")

# Show impact of outlier
mae_increase = outlier_eval.MAE() / normal_eval.MAE()
rmse_increase = outlier_eval.RMSE() / normal_eval.RMSE()

print(f"\nOutlier Impact:")
print(f"  MAE increased by: {mae_increase:.1f}x")
print(f"  RMSE increased by: {rmse_increase:.1f}x")
print(f"  RMSE is more sensitive to outliers!")
```

### R-squared (Coefficient of Determination)

**Description**: Proportion of variance in the target variable explained by the model.

**Formula**: `R² = 1 - (SS_res / SS_tot)`

**Range**: -∞ to 1 (higher is better, 1 is perfect)

**Best For**: Understanding how well the model explains the data

```python
from jackofalltrades.Errors import Error
import numpy as np
import matplotlib.pyplot as plt

# Generate datasets with different R² values
np.random.seed(42)
x = np.linspace(0, 10, 100)

# Perfect linear relationship (R² ≈ 1)
y_perfect = 2 * x + 1
y_pred_perfect = y_perfect  # Perfect predictions

# Good linear relationship (R² ≈ 0.8)
y_good = 2 * x + 1 + np.random.normal(0, 2, 100)  # Add noise
y_pred_good = 2 * x + 1  # True relationship

# Poor relationship (R² ≈ 0.3)
y_poor = 2 * x + 1 + np.random.normal(0, 5, 100)  # More noise
y_pred_poor = 2 * x + 1  # True relationship

# Random predictions (R² ≈ 0)
y_random = 2 * x + 1 + np.random.normal(0, 1, 100)
y_pred_random = np.random.normal(10, 5, 100)  # Random predictions

# Calculate R² for each case
perfect_r2 = Error(y_true=y_perfect, y_predicted=y_pred_perfect).RSquared()
good_r2 = Error(y_true=y_good, y_predicted=y_pred_good).RSquared()
poor_r2 = Error(y_true=y_poor, y_predicted=y_pred_poor).RSquared()
random_r2 = Error(y_true=y_random, y_predicted=y_pred_random).RSquared()

print("R² Examples:")
print(f"Perfect predictions: R² = {perfect_r2:.3f}")
print(f"Good predictions: R² = {good_r2:.3f}")
print(f"Poor predictions: R² = {poor_r2:.3f}")
print(f"Random predictions: R² = {random_r2:.3f}")

# R² interpretation guide
print(f"\nR² Interpretation Guide:")
print(f"  R² > 0.9: Excellent model")
print(f"  0.7 < R² ≤ 0.9: Good model")
print(f"  0.5 < R² ≤ 0.7: Moderate model")
print(f"  0.3 < R² ≤ 0.5: Weak model")
print(f"  R² ≤ 0.3: Poor model")
print(f"  R² < 0: Worse than predicting the mean")
```

---

## 🎯 Classification Metrics

### Detailed Classification Analysis

```python
from jackofalltrades.Errors import accuracy, f1score
import numpy as np

def classification_report_detailed(y_true, y_pred, class_names=None):
    """
    Comprehensive classification analysis
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall metrics
    overall_acc = accuracy(y_true, y_pred)
    overall_f1 = f1score(y_true, y_pred)
    
    print("=== Classification Report ===")
    print(f"Overall Accuracy: {overall_acc:.3f}")
    print(f"Overall F1 Score: {overall_f1:.3f}")
    
    # Confusion matrix
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)
    
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i, true_class in enumerate(unique_classes):
        for j, pred_class in enumerate(unique_classes):
            confusion_matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    print(f"\nConfusion Matrix:")
    print("Actual\\Predicted", end="")
    for class_label in unique_classes:
        print(f"\t{class_label}", end="")
    print()
    
    for i, true_class in enumerate(unique_classes):
        print(f"{true_class}\t\t", end="")
        for j in range(n_classes):
            print(f"{confusion_matrix[i, j]}\t", end="")
        print()
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"Class\tPrecision\tRecall\t\tF1-Score\tSupport")
    
    for class_label in unique_classes:
        # Binary classification for this class
        y_binary_true = (y_true == class_label).astype(int)
        y_binary_pred = (y_pred == class_label).astype(int)
        
        tp = np.sum((y_binary_true == 1) & (y_binary_pred == 1))
        fp = np.sum((y_binary_true == 0) & (y_binary_pred == 1))
        fn = np.sum((y_binary_true == 1) & (y_binary_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == class_label)
        
        print(f"{class_label}\t{precision:.3f}\t\t{recall:.3f}\t\t{f1:.3f}\t\t{support}")
    
    return confusion_matrix

# Example usage
np.random.seed(42)

# Simulate multi-class classification results
y_true_multi = np.random.choice([0, 1, 2], 100, p=[0.5, 0.3, 0.2])  # Imbalanced classes
y_pred_multi = y_true_multi.copy()

# Add some classification errors
error_indices = np.random.choice(100, 20, replace=False)
y_pred_multi[error_indices] = np.random.choice([0, 1, 2], 20)

# Generate detailed report
confusion_mat = classification_report_detailed(y_true_multi, y_pred_multi)
```

---

## 🔄 Advanced Evaluation

### Cross-Validation with jackofalltrades

```python
from jackofalltrades.Errors import Error, accuracy, f1score
from jackofalltrades.datasets import get_real_estate
from jackofalltrades.Models import LinearRegression, RidgeRegression
from sklearn.model_selection import KFold
import numpy as np

def cross_validate_regression(X, y, model, cv=5, metrics=['mse', 'rmse', 'r2']):
    """
    Cross-validation for regression models using jackofalltrades
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    results = {metric: [] for metric in metrics}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_val)
        
        # Evaluate
        evaluator = Error(y_true=y_val, y_predicted=predictions)
        
        # Store metrics
        if 'mse' in metrics:
            results['mse'].append(evaluator.MSE())
        if 'rmse' in metrics:
            results['rmse'].append(evaluator.RMSE())
        if 'mae' in metrics:
            results['mae'].append(evaluator.MAE())
        if 'r2' in metrics:
            results['r2'].append(evaluator.RSquared())
    
    # Print results
    print(f"Cross-Validation Results ({cv} folds):")
    for metric, values in results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    return results

# Example usage
X, y = get_real_estate()

# Compare models using cross-validation
print("Linear Regression:")
linear_results = cross_validate_regression(X, y, LinearRegression())

print("\nRidge Regression:")
ridge_results = cross_validate_regression(X, y, RidgeRegression())

# Statistical comparison
print(f"\nModel Comparison:")
for metric in ['mse', 'rmse', 'r2']:
    linear_mean = np.mean(linear_results[metric])
    ridge_mean = np.mean(ridge_results[metric])
    
    if metric in ['mse', 'rmse']:  # Lower is better
        better = "Ridge" if ridge_mean < linear_mean else "Linear"
        diff = abs(ridge_mean - linear_mean) / linear_mean * 100
    else:  # Higher is better (R²)
        better = "Ridge" if ridge_mean > linear_mean else "Linear"
        diff = abs(ridge_mean - linear_mean) / linear_mean * 100
    
    print(f"  {metric.upper()}: {better} is {diff:.1f}% better")
```

---

## 🎯 Best Practices

### 1. Choose the Right Metric

```python
def metric_selection_guide():
    """
    Guide for selecting appropriate metrics
    """
    print("=== Metric Selection Guide ===\n")
    
    print("REGRESSION METRICS:")
    print("  MSE: Use when large errors are particularly bad")
    print("  RMSE: Use when you want interpretable units")
    print("  MAE: Use when outliers shouldn't dominate")
    print("  R²: Use to understand explained variance")
    
    print("\nCLASSIFICATION METRICS:")
    print("  Accuracy: Use for balanced datasets")
    print("  F1 Score: Use for imbalanced datasets")
    print("  Precision: Use when false positives are costly")
    print("  Recall: Use when false negatives are costly")
    
    print("\nCOMMON SCENARIOS:")
    print("  Medical diagnosis: High recall (don't miss diseases)")
    print("  Spam detection: High precision (don't flag important emails)")
    print("  House prices: RMSE (interpretable dollar amounts)")
    print("  Model comparison: R² (easy to compare explained variance)")

metric_selection_guide()
```

### 2. Validation Strategies

```python
def validation_strategies():
    """
    Different validation approaches for different scenarios
    """
    print("=== Validation Strategies ===\n")
    
    strategies = {
        "Random Split": {
            "use_case": "General purpose, IID data",
            "split": "70-80% train, 20-30% test",
            "example": "House prices, customer classification"
        },
        "Time Series Split": {
            "use_case": "Time-dependent data",
            "split": "Earlier data for training, later for testing",
            "example": "Stock prices, weather forecasting"
        },
        "Stratified Split": {
            "use_case": "Imbalanced classification",
            "split": "Maintain class proportions in train/test",
            "example": "Rare disease detection, fraud detection"
        },
        "Cross-Validation": {
            "use_case": "Small datasets, robust evaluation",
            "split": "Multiple train/validation splits",
            "example": "Medical studies, A/B testing"
        }
    }
    
    for strategy, details in strategies.items():
        print(f"{strategy}:")
        print(f"  Use case: {details['use_case']}")
        print(f"  Split: {details['split']}")
        print(f"  Example: {details['example']}\n")

validation_strategies()
```

---

## 🚀 Next Steps

1. **Practice with different datasets** from [Datasets Reference](datasets-reference.md)
2. **Try various models** from [Models Guide](models_guide.md)
3. **Implement cross-validation** for robust evaluation
4. **Analyze residuals** to understand model limitations
5. **Compare multiple metrics** to get a complete picture

For more information, see:
- [Models Guide](models_guide.md) - Learn about available models
- [Datasets Reference](datasets-reference.md) - Practice with different datasets
- [API Documentation](api_reference.md) - Detailed function references 