# 📊 Datasets Guide - jackofalltrades

This comprehensive guide covers all datasets available in the `jackofalltrades` library. Each dataset is carefully curated and ready-to-use for machine learning experiments, learning, and prototyping.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Real Estate Datasets](#real-estate-datasets)
3. [Financial Datasets](#financial-datasets)
4. [Available Datasets Explorer](#available-datasets-explorer)
5. [Dataset Comparison](#dataset-comparison)
6. [Advanced Usage](#advanced-usage)

---

## 🚀 Quick Start

All datasets in `jackofalltrades` follow a simple, consistent API:

```python
from jackofalltrades.datasets import get_dataset_name

# Load dataset
X, y = get_dataset_name()

# X contains the features (input variables)  
# y contains the target variable (what we want to predict)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {X.columns.tolist()}")
```

---

## 🏠 Real Estate Datasets

Perfect for learning regression and understanding real-world prediction problems.

### Real Estate Dataset

**Description**: Property pricing data with location, size, and amenity features.

**Use Cases**:
- House price prediction
- Real estate market analysis
- Feature importance in property valuation
- Regression model training

**Target Variable**: Property price

```python
from jackofalltrades.datasets import get_real_estate
from jackofalltrades.Models import LinearRegression
from jackofalltrades.Errors import Error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
X, y = get_real_estate()

print("Real Estate Dataset Overview:")
print(f"Dataset shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"\nFeature columns:")
for i, col in enumerate(X.columns):
    print(f"  {i+1}. {col}")

print(f"\nFirst few rows:")
print(X.head())

print(f"\nTarget statistics:")
print(f"  Mean price: ${y.mean():,.2f}")
print(f"  Min price: ${y.min():,.2f}")
print(f"  Max price: ${y.max():,.2f}")
print(f"  Std deviation: ${y.std():,.2f}")

# Basic analysis
print(f"\nDataset Info:")
print(f"  Number of properties: {len(X)}")
print(f"  Number of features: {X.shape[1]}")
print(f"  Missing values: {X.isnull().sum().sum()}")

# Train a quick model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

evaluator = Error(y_true=y, y_predicted=predictions)
print(f"\nQuick Model Performance:")
print(f"  R² Score: {evaluator.RSquared():.4f}")
print(f"  RMSE: ${evaluator.RMSE():,.2f}")

# Visualize price distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(y, bins=30, alpha=0.7, edgecolor='black')
plt.title('Property Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.scatter(y, predictions, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')

plt.tight_layout()
plt.show()
```

**Advanced Analysis Example**:

```python
import seaborn as sns
import numpy as np

# Load data
X, y = get_real_estate()

# Feature correlation analysis
correlation_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Feature importance (correlation with target)
feature_importance = X.corrwith(y).abs().sort_values(ascending=False)
print("Feature Importance (Correlation with Price):")
for feature, importance in feature_importance.items():
    print(f"  {feature}: {importance:.3f}")

# Price prediction for different scenarios
scenarios = {
    'Small Apartment': X.iloc[0].copy(),
    'Large House': X.iloc[0].copy(),
    'Luxury Property': X.iloc[0].copy()
}

# Modify scenarios (this is conceptual - actual modifications depend on feature meanings)
print("\nPrice Predictions for Different Scenarios:")
model = LinearRegression()
model.fit(X, y)

for scenario_name, scenario_data in scenarios.items():
    pred_price = model.predict(scenario_data.values.reshape(1, -1))[0]
    print(f"  {scenario_name}: ${pred_price:,.2f}")
```

---

## 💰 Financial Datasets

Essential for financial modeling and time series analysis.

### California Housing Dataset

**Description**: Housing data from California census with geographic and demographic features.

**Use Cases**:
- Regional price analysis
- Geographic modeling
- Feature engineering practice
- Regression benchmarking

```python
from jackofalltrades.datasets import get_california_housing
import pandas as pd
import numpy as np

# Load dataset
X, y = get_california_housing()

print("California Housing Dataset Overview:")
print(f"Dataset shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

print(f"\nDataset statistics:")
print(X.describe())

print(f"\nTarget variable (housing prices):")
print(f"  Mean: ${y.mean():,.2f}")
print(f"  Median: ${y.median():,.2f}")
print(f"  Range: ${y.min():,.2f} - ${y.max():,.2f}")

# Geographic analysis (if location features available)
if 'longitude' in X.columns and 'latitude' in X.columns:
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X['longitude'], X['latitude'], 
                         c=y, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, label='House Price')
    plt.title('California Housing Prices by Location')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Feature distribution analysis
plt.figure(figsize=(15, 10))
for i, column in enumerate(X.columns[:6]):  # First 6 features
    plt.subplot(2, 3, i+1)
    plt.hist(X[column], bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Bitcoin Dataset

**Description**: Bitcoin price and trading data with technical indicators.

**Use Cases**:
- Cryptocurrency price prediction
- Time series forecasting
- Financial trend analysis
- Volatility modeling

```python
from jackofalltrades.datasets import get_bitcoin
import matplotlib.pyplot as plt
import pandas as pd

# Load Bitcoin dataset
X, y = get_bitcoin()

print("Bitcoin Dataset Overview:")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features: {X.columns.tolist()}")

print(f"\nBitcoin price statistics:")
print(f"  Mean price: ${y.mean():,.2f}")
print(f"  Min price: ${y.min():,.2f}")
print(f"  Max price: ${y.max():,.2f}")
print(f"  Volatility (std): ${y.std():,.2f}")

# Time series visualization
plt.figure(figsize=(15, 10))

# Price over time
plt.subplot(2, 2, 1)
plt.plot(range(len(y)), y, linewidth=1)
plt.title('Bitcoin Price Over Time')
plt.xlabel('Time Period')
plt.ylabel('Price ($)')

# Price distribution
plt.subplot(2, 2, 2)
plt.hist(y, bins=50, alpha=0.7, edgecolor='black')
plt.title('Bitcoin Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')

# Returns calculation and visualization
returns = np.diff(y) / y[:-1] * 100  # Percentage returns
plt.subplot(2, 2, 3)
plt.plot(range(len(returns)), returns, linewidth=0.5)
plt.title('Bitcoin Daily Returns (%)')
plt.xlabel('Time Period')
plt.ylabel('Return (%)')

# Returns distribution
plt.subplot(2, 2, 4)
plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
plt.title('Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Volatility analysis
print(f"\nVolatility Analysis:")
print(f"  Daily return mean: {returns.mean():.3f}%")
print(f"  Daily return std: {returns.std():.3f}%")
print(f"  Max daily gain: {returns.max():.2f}%")
print(f"  Max daily loss: {returns.min():.2f}%")

# Technical analysis example
window_size = 20
if len(y) > window_size:
    moving_avg = pd.Series(y).rolling(window=window_size).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y)), y, label='Price', alpha=0.7)
    plt.plot(range(len(moving_avg)), moving_avg, label=f'{window_size}-day MA', linewidth=2)
    plt.title('Bitcoin Price with Moving Average')
    plt.xlabel('Time Period')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()
```

---

## 🔍 Available Datasets Explorer

The `jackofalltrades` library includes several datasets. Here's how to explore all available datasets:

```python
# Explore all available dataset functions
import jackofalltrades.datasets as datasets
import inspect

print("Available Dataset Functions:")
dataset_functions = [name for name, obj in inspect.getmembers(datasets) 
                    if inspect.isfunction(obj) and name.startswith('get_')]

for i, func_name in enumerate(dataset_functions, 1):
    print(f"  {i}. {func_name}()")

# Load each dataset to see its characteristics
print("\nDataset Overview:")
for func_name in dataset_functions:
    try:
        func = getattr(datasets, func_name)
        X, y = func()
        print(f"\n{func_name}:")
        print(f"  Features: {X.shape}")
        print(f"  Target: {y.shape}")
        if hasattr(X, 'columns'):
            print(f"  Feature names: {list(X.columns)[:5]}...")  # First 5 features
        print(f"  Target type: {type(y.iloc[0] if hasattr(y, 'iloc') else y[0])}")
    except Exception as e:
        print(f"\n{func_name}: Error loading - {str(e)}")
```

### Complete Dataset Exploration

```python
from jackofalltrades.datasets import *
import pandas as pd
import numpy as np

def explore_all_datasets():
    """
    Systematically explore all available datasets
    """
    # List of known dataset functions
    dataset_loaders = [
        get_real_estate,
        get_california_housing,
        get_bitcoin,
        # Add more as they become available
    ]
    
    results = []
    
    for loader in dataset_loaders:
        try:
            print(f"\n{'='*50}")
            print(f"Exploring: {loader.__name__}")
            print(f"{'='*50}")
            
            # Load dataset
            X, y = loader()
            
            # Basic info
            info = {
                'name': loader.__name__,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'target_type': 'continuous' if np.issubdtype(y.dtype, np.number) else 'categorical',
                'has_missing': X.isnull().sum().sum() > 0,
                'feature_types': X.dtypes.value_counts().to_dict()
            }
            
            print(f"Dataset: {info['name']}")
            print(f"  Samples: {info['n_samples']}")
            print(f"  Features: {info['n_features']}")
            print(f"  Target type: {info['target_type']}")
            print(f"  Missing values: {info['has_missing']}")
            print(f"  Feature types: {info['feature_types']}")
            
            # Feature names
            if hasattr(X, 'columns'):
                print(f"  Feature names: {list(X.columns)}")
            
            # Target statistics
            if info['target_type'] == 'continuous':
                print(f"  Target range: {y.min():.2f} to {y.max():.2f}")
                print(f"  Target mean: {y.mean():.2f}")
                print(f"  Target std: {y.std():.2f}")
            else:
                print(f"  Target classes: {np.unique(y)}")
                print(f"  Class distribution: {np.bincount(y)}")
            
            results.append(info)
            
        except Exception as e:
            print(f"Error with {loader.__name__}: {str(e)}")
    
    return results

# Run exploration
dataset_info = explore_all_datasets()
```

---

## 📊 Dataset Comparison

### Dataset Characteristics Summary

| Dataset | Size | Features | Target | Best For | Difficulty |
|---------|------|----------|--------|----------|------------|
| **Real Estate** | Medium | Mixed | Continuous | Regression Learning | Beginner |
| **California Housing** | Large | Numeric | Continuous | Geographic Analysis | Intermediate |
| **Bitcoin** | Variable | Time Series | Continuous | Financial Modeling | Advanced |

### Choosing the Right Dataset

**For Beginners**:
```python
# Start with real estate - clean and interpretable
from jackofalltrades.datasets import get_real_estate
X, y = get_real_estate()
```

**For Intermediate Users**:
```python
# Try California housing for more complexity
from jackofalltrades.datasets import get_california_housing
X, y = get_california_housing()
```

**For Advanced Users**:
```python
# Bitcoin for time series and financial modeling
from jackofalltrades.datasets import get_bitcoin
X, y = get_bitcoin()
```

### Dataset Selection Guide

```python
def recommend_dataset(goal, experience_level):
    """
    Recommend datasets based on learning goals and experience
    """
    recommendations = {
        ('regression', 'beginner'): {
            'dataset': 'get_real_estate',
            'reason': 'Clean data, interpretable features, good for learning basics'
        },
        ('regression', 'intermediate'): {
            'dataset': 'get_california_housing',
            'reason': 'More features, geographic data, good for feature engineering'
        },
        ('regression', 'advanced'): {
            'dataset': 'get_bitcoin',
            'reason': 'Time series, financial data, volatility modeling'
        },
        ('classification', 'beginner'): {
            'dataset': 'get_real_estate (binary)',
            'reason': 'Convert to binary classification (expensive vs cheap)'
        },
        ('time_series', 'any'): {
            'dataset': 'get_bitcoin',
            'reason': 'Sequential data with temporal dependencies'
        }
    }
    
    key = (goal, experience_level)
    if key in recommendations:
        rec = recommendations[key]
        print(f"Recommended: {rec['dataset']}")
        print(f"Reason: {rec['reason']}")
    else:
        print("Try get_real_estate() for a good starting point!")

# Example usage
recommend_dataset('regression', 'beginner')
recommend_dataset('regression', 'advanced')
recommend_dataset('time_series', 'any')
```

---

## 🔧 Advanced Usage

### Custom Data Preprocessing

```python
from jackofalltrades.datasets import get_real_estate
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Load dataset
X, y = get_real_estate()

# Custom preprocessing function
def preprocess_dataset(X, y, scale_features=True, handle_categorical=True):
    """
    Advanced preprocessing for jackofalltrades datasets
    """
    X_processed = X.copy()
    
    # Handle categorical variables
    if handle_categorical:
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
    
    # Scale numerical features
    if scale_features:
        numerical_columns = X_processed.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        X_processed[numerical_columns] = scaler.fit_transform(X_processed[numerical_columns])
    
    return X_processed, y

# Apply preprocessing
X_processed, y_processed = preprocess_dataset(X, y)

print("Preprocessing Results:")
print(f"Original shape: {X.shape}")
print(f"Processed shape: {X_processed.shape}")
print(f"Data types after processing:")
print(X_processed.dtypes)
```

### Feature Engineering

```python
import numpy as np
import pandas as pd

def create_additional_features(X, y):
    """
    Create additional features for better model performance
    """
    X_enhanced = X.copy()
    
    # Polynomial features (for first 3 numeric columns)
    numeric_cols = X_enhanced.select_dtypes(include=[np.number]).columns[:3]
    for col in numeric_cols:
        X_enhanced[f'{col}_squared'] = X_enhanced[col] ** 2
        X_enhanced[f'{col}_sqrt'] = np.sqrt(np.abs(X_enhanced[col]))
    
    # Interaction features
    if len(numeric_cols) >= 2:
        X_enhanced[f'{numeric_cols[0]}_x_{numeric_cols[1]}'] = (
            X_enhanced[numeric_cols[0]] * X_enhanced[numeric_cols[1]]
        )
    
    # Binning features
    for col in numeric_cols:
        X_enhanced[f'{col}_binned'] = pd.cut(X_enhanced[col], bins=5, labels=False)
    
    return X_enhanced

# Example usage
X, y = get_real_estate()
X_enhanced = create_additional_features(X, y)

print(f"Original features: {X.shape[1]}")
print(f"Enhanced features: {X_enhanced.shape[1]}")
print(f"New features added: {X_enhanced.shape[1] - X.shape[1]}")
```

### Data Splitting Strategies

```python
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
import numpy as np

def split_dataset(X, y, strategy='random', test_size=0.2, random_state=42):
    """
    Multiple data splitting strategies for different use cases
    """
    
    if strategy == 'random':
        # Standard random split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    elif strategy == 'time_series':
        # Time-based split (for time series data like Bitcoin)
        split_idx = int((1 - test_size) * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test
    
    elif strategy == 'stratified':
        # Stratified split (for classification tasks)
        return train_test_split(X, y, test_size=test_size, 
                              stratify=y, random_state=random_state)

# Example usage with different datasets
datasets = [
    ('Real Estate', get_real_estate),
    ('California Housing', get_california_housing),
    ('Bitcoin', get_bitcoin)
]

for name, loader in datasets:
    X, y = loader()
    
    # Choose appropriate split strategy
    if 'Bitcoin' in name:
        strategy = 'time_series'
    else:
        strategy = 'random'
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, strategy=strategy)
    
    print(f"\n{name} Dataset Split ({strategy}):")
    print(f"  Training: {X_train.shape}")
    print(f"  Testing: {X_test.shape}")
```

### Dataset Validation

```python
def validate_dataset(X, y, dataset_name="Dataset"):
    """
    Comprehensive dataset validation
    """
    print(f"\n=== {dataset_name} Validation ===")
    
    # Basic checks
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Sample alignment: {len(X) == len(y)}")
    
    # Missing values
    missing_features = X.isnull().sum()
    missing_target = y.isnull().sum() if hasattr(y, 'isnull') else 0
    
    print(f"✓ Missing values in features: {missing_features.sum()}")
    print(f"✓ Missing values in target: {missing_target}")
    
    # Data types
    if hasattr(X, 'dtypes'):
        print(f"✓ Feature data types: {X.dtypes.value_counts().to_dict()}")
    
    # Target distribution
    if hasattr(y, 'dtype'):
        if y.dtype in ['object', 'category']:
            print(f"✓ Target classes: {y.value_counts().to_dict()}")
        else:
            print(f"✓ Target range: {y.min():.2f} to {y.max():.2f}")
            print(f"✓ Target mean: {y.mean():.2f}")
    
    # Potential issues
    issues = []
    if missing_features.sum() > 0:
        issues.append("Missing values in features")
    if missing_target > 0:
        issues.append("Missing values in target")
    if len(X) < 100:
        issues.append("Small dataset size")
    
    if issues:
        print(f"⚠️  Potential issues: {', '.join(issues)}")
    else:
        print("✅ No major issues detected")

# Validate all datasets
datasets = [
    ('Real Estate', get_real_estate),
    ('California Housing', get_california_housing),
    ('Bitcoin', get_bitcoin)
]

for name, loader in datasets:
    try:
        X, y = loader()
        validate_dataset(X, y, name)
    except Exception as e:
        print(f"\n=== {name} Validation ===")
        print(f"❌ Error loading dataset: {str(e)}")
```

---

## 🎯 Best Practices

1. **Always explore your data first**:
   ```python
   X, y = get_dataset()
   print(X.info())
   print(X.describe())
   ```

2. **Check for missing values**:
   ```python
   print(X.isnull().sum())
   ```

3. **Visualize distributions**:
   ```python
   import matplotlib.pyplot as plt
   X.hist(bins=20, figsize=(12, 8))
   plt.show()
   ```

4. **Split data properly**:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

5. **Scale features for neural networks**:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

---

## 🚀 Next Steps

1. **Choose a dataset** that matches your learning goals
2. **Explore the data** thoroughly before modeling
3. **Try different models** from the [Models Guide](models_guide.md)
4. **Evaluate performance** using [Evaluation Metrics](evaluation-metrics.md)
5. **Experiment** with different preprocessing techniques

For more information, see:
- [Models Guide](models_guide.md) - Learn about available models
- [Evaluation Metrics](evaluation-metrics.md) - Understand model evaluation
- [API Documentation](api_reference.md) - Detailed function references 