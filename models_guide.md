# 🧠 Models Guide - jackofalltrades

This comprehensive guide covers all machine learning models available in the `jackofalltrades` library. Each model is designed with simplicity and effectiveness in mind, making them perfect for both beginners and experienced practitioners.

## 📋 Table of Contents

1. [Regression Models](#regression-models)
   - [Linear Regression](#linear-regression)
   - [Ridge Regression](#ridge-regression)
   - [Adaptive Regression](#adaptive-regression)
   - [MLP Regressor](#mlp-regressor)
2. [Classification Models](#classification-models)
   - [Logistic Regression](#logistic-regression)
   - [Image Classification](#image-classification)
3. [Generative Models](#generative-models)
   - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
   - [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)

---

## 🔢 Regression Models

Regression models predict continuous numerical values. Use these when your target variable is a number that can take any value within a range.

### Linear Regression

**Use Case**: Predicting continuous values with a linear relationship between features and target.

**Best For**:
- House price prediction
- Sales forecasting
- Simple trend analysis
- When interpretability is crucial

**Example 1: House Price Prediction**

```python
from jackofalltrades.datasets import get_real_estate
from jackofalltrades.Models import LinearRegression
from jackofalltrades.Errors import Error
import matplotlib.pyplot as plt

# Load real estate dataset
X, y = get_real_estate()
print(f"Dataset shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate performance
evaluator = Error(y_true=y, y_predicted=predictions)
print(f"Model Performance:")
print(f"R² Score: {evaluator.RSquared():.4f}")
print(f"MSE: {evaluator.MSE():.4f}")
print(f"RMSE: {evaluator.RMSE():.4f}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted House Prices')
plt.show()
```

**Example 2: Custom Dataset**

```python
import numpy as np
from jackofalltrades.Models import LinearRegression

# Create synthetic data
np.random.seed(42)
X = np.random.randn(1000, 3)  # 3 features
true_weights = [2.5, -1.3, 0.8]
y = X @ true_weights + 0.1 * np.random.randn(1000)  # Add noise

# Train model
model = LinearRegression()
model.fit(X, y)

# Get model coefficients (if available)
predictions = model.predict(X)
print(f"Predictions shape: {predictions.shape}")
```

---

### Ridge Regression

**Use Case**: Linear regression with L2 regularization to prevent overfitting.

**Best For**:
- When you have many features
- Preventing overfitting
- Feature multicollinearity issues
- More stable predictions

**Example: California Housing with Regularization**

```python
from jackofalltrades.datasets import get_california_housing
from jackofalltrades.Models import RidgeRegression, LinearRegression
from jackofalltrades.Errors import Error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
X, y = get_california_housing()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge regression
ridge_model = RidgeRegression()
ridge_model.fit(X_train_scaled, y_train)

# Compare with Linear regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predictions
ridge_pred = ridge_model.predict(X_test_scaled)
linear_pred = linear_model.predict(X_test_scaled)

# Evaluate both models
ridge_eval = Error(y_true=y_test, y_predicted=ridge_pred)
linear_eval = Error(y_true=y_test, y_predicted=linear_pred)

print("Model Comparison:")
print(f"Ridge Regression - R²: {ridge_eval.RSquared():.4f}, RMSE: {ridge_eval.RMSE():.4f}")
print(f"Linear Regression - R²: {linear_eval.RSquared():.4f}, RMSE: {linear_eval.RMSE():.4f}")
```

---

### Adaptive Regression

**Use Case**: Advanced regression that adapts to data patterns automatically.

**Best For**:
- Complex, non-linear relationships
- When simple linear models fail
- Automatic feature engineering
- Robust predictions

**Example: Bitcoin Price Prediction**

```python
from jackofalltrades.datasets import get_bitcoin
from jackofalltrades.Models import AdaptiveRegression
from jackofalltrades.Errors import Error
import pandas as pd

# Load Bitcoin dataset
X, y = get_bitcoin()
print(f"Bitcoin dataset loaded: {X.shape}")

# Create time-based train/test split (important for time series)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train adaptive regression
adaptive_model = AdaptiveRegression()
adaptive_model.fit(X_train, y_train)

# Make predictions
train_pred = adaptive_model.predict(X_train)
test_pred = adaptive_model.predict(X_test)

# Evaluate performance
train_eval = Error(y_true=y_train, y_predicted=train_pred)
test_eval = Error(y_true=y_test, y_predicted=test_pred)

print("Adaptive Regression Results:")
print(f"Train R²: {train_eval.RSquared():.4f}")
print(f"Test R²: {test_eval.RSquared():.4f}")
print(f"Test RMSE: {test_eval.RMSE():.4f}")

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.plot(range(len(y_train)), y_train, label='Training Actual', alpha=0.7)
plt.plot(range(len(y_train)), train_pred, label='Training Predicted', alpha=0.7)
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Actual')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), test_pred, label='Test Predicted')
plt.axvline(x=len(y_train), color='red', linestyle='--', alpha=0.7, label='Train/Test Split')
plt.legend()
plt.title('Bitcoin Price Prediction with Adaptive Regression')
plt.show()
```

---

### MLP Regressor

**Use Case**: Multi-layer perceptron (neural network) for complex non-linear regression.

**Best For**:
- Complex, non-linear patterns
- High-dimensional data
- When you have plenty of training data
- Feature interactions matter

**Example: Advanced Pattern Recognition**

```python
from jackofalltrades.Models import MLPRegressor, LinearRegression
from jackofalltrades.Errors import Error
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate complex synthetic dataset
X, y = make_regression(
    n_samples=2000, 
    n_features=10, 
    n_informative=8,
    noise=0.1, 
    random_state=42
)

# Add non-linear transformations to make it more complex
X_complex = np.column_stack([
    X,
    X[:, 0] * X[:, 1],  # Interaction terms
    np.sin(X[:, 2]),     # Non-linear transformations
    X[:, 3] ** 2
])

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X_complex, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP Regressor
mlp_model = MLPRegressor()
mlp_model.fit(X_train_scaled, y_train)

# Compare with Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predictions
mlp_pred = mlp_model.predict(X_test_scaled)
linear_pred = linear_model.predict(X_test_scaled)

# Evaluation
mlp_eval = Error(y_true=y_test, y_predicted=mlp_pred)
linear_eval = Error(y_true=y_test, y_predicted=linear_pred)

print("Complex Pattern Recognition Results:")
print(f"MLP Regressor - R²: {mlp_eval.RSquared():.4f}, RMSE: {mlp_eval.RMSE():.4f}")
print(f"Linear Baseline - R²: {linear_eval.RSquared():.4f}, RMSE: {linear_eval.RMSE():.4f}")
print(f"Improvement: {mlp_eval.RSquared() - linear_eval.RSquared():.4f}")
```

---

## 🎯 Classification Models

Classification models predict discrete categories or classes. Use these when your target variable represents categories.

### Logistic Regression

**Use Case**: Binary and multi-class classification with interpretable results.

**Best For**:
- Binary classification (yes/no, spam/not spam)
- When you need probability estimates
- Interpretable models
- Baseline classification model

**Example 1: Binary Classification**

```python
from jackofalltrades.Models import LogisticRegression
from jackofalltrades.Errors import accuracy, f1score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Generate binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_redundant=0,
    n_informative=8,
    n_clusters_per_class=1,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train logistic regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Make predictions
train_pred = log_model.predict(X_train)
test_pred = log_model.predict(X_test)

# Evaluate performance
train_acc = accuracy(y_train, train_pred)
test_acc = accuracy(y_test, test_pred)
test_f1 = f1score(y_test, test_pred)

print("Logistic Regression Results:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

# Class distribution
unique, counts = np.unique(y_test, return_counts=True)
print(f"Test set class distribution: {dict(zip(unique, counts))}")
```

**Example 2: Real-world Application - Customer Churn**

```python
import pandas as pd
import numpy as np
from jackofalltrades.Models import LogisticRegression

# Simulate customer data
np.random.seed(42)
n_customers = 1000

# Generate features
data = {
    'tenure_months': np.random.randint(1, 73, n_customers),
    'monthly_charges': np.random.normal(65, 20, n_customers),
    'total_charges': np.random.normal(2300, 1000, n_customers),
    'contract_type': np.random.choice([0, 1, 2], n_customers),  # 0: month-to-month, 1: 1-year, 2: 2-year
    'payment_method': np.random.choice([0, 1, 2, 3], n_customers),
    'tech_support': np.random.choice([0, 1], n_customers),
}

df = pd.DataFrame(data)

# Create target variable (churn) based on logical rules
churn_prob = (
    0.1 + 
    0.3 * (df['tenure_months'] < 12) +  # New customers more likely to churn
    0.2 * (df['contract_type'] == 0) +   # Month-to-month more likely
    0.1 * (df['tech_support'] == 0) -    # No tech support increases churn
    0.01 * df['tenure_months']           # Longer tenure decreases churn
)

y = np.random.binomial(1, churn_prob, n_customers)

# Prepare features
X = df.values

# Train model
churn_model = LogisticRegression()
churn_model.fit(X, y)

# Predict churn for new customers
new_customers = np.array([
    [6, 85, 510, 0, 2, 0],   # High-risk customer
    [36, 55, 1980, 2, 1, 1], # Low-risk customer
])

churn_predictions = churn_model.predict(new_customers)
print("Churn Predictions for New Customers:")
print(f"Customer 1 (high-risk): {'Will Churn' if churn_predictions[0] else 'Will Stay'}")
print(f"Customer 2 (low-risk): {'Will Churn' if churn_predictions[1] else 'Will Stay'}")
```

---

### Image Classification

**Use Case**: Deep learning model for classifying images.

**Best For**:
- Computer vision tasks
- Image recognition
- MNIST, CIFAR-10 type problems
- Any visual classification task

**Example 1: MNIST-style Digit Classification**

```python
from jackofalltrades.Models import ImageClassification
from jackofalltrades.Errors import accuracy
import numpy as np
import matplotlib.pyplot as plt

# Simulate MNIST-like data (28x28 grayscale images)
np.random.seed(42)
n_samples = 1000
n_classes = 10

# Generate synthetic image data
X = np.random.rand(n_samples, 28, 28, 1)  # Random noise images
y = np.random.randint(0, n_classes, n_samples)  # Random labels

# Add some pattern to make it learnable
for i in range(n_samples):
    digit = y[i]
    # Add digit-specific patterns
    X[i, 10:18, 10:18, 0] = digit * 0.1  # Center square with digit intensity
    X[i, digit*2:(digit*2)+5, digit*2:(digit*2)+5, 0] = 0.8  # Diagonal pattern

# Split data
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Number of classes: {n_classes}")

# Initialize and train model
image_model = ImageClassification(
    input_shape=(28, 28, 1), 
    num_classes=n_classes
)

# Train the model
image_model.fit(X_train, y_train)

# Make predictions
train_pred = image_model.predict(X_train)
test_pred = image_model.predict(X_test)

# Evaluate
train_acc = accuracy(y_train, train_pred)
test_acc = accuracy(y_test, test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(X_test[i].squeeze(), cmap='gray')
    axes[row, col].set_title(f'True: {y_test[i]}, Pred: {test_pred[i]}')
    axes[row, col].axis('off')
plt.suptitle('Image Classification Results')
plt.show()
```

**Example 2: Custom Image Data**

```python
from jackofalltrades.Models import ImageClassification
import numpy as np

# Simulate RGB color images (32x32)
n_samples = 500
X_rgb = np.random.rand(n_samples, 32, 32, 3)  # RGB images
y_rgb = np.random.randint(0, 5, n_samples)     # 5 classes

# Add color-based patterns
for i in range(n_samples):
    class_label = y_rgb[i]
    # Each class has a dominant color channel
    if class_label == 0:  # Red dominant
        X_rgb[i, :, :, 0] += 0.5
    elif class_label == 1:  # Green dominant
        X_rgb[i, :, :, 1] += 0.5
    elif class_label == 2:  # Blue dominant
        X_rgb[i, :, :, 2] += 0.5
    elif class_label == 3:  # Yellow (Red + Green)
        X_rgb[i, :, :, 0] += 0.3
        X_rgb[i, :, :, 1] += 0.3
    # Class 4 remains random

# Clip values to [0, 1]
X_rgb = np.clip(X_rgb, 0, 1)

# Train model for RGB images
rgb_model = ImageClassification(
    input_shape=(32, 32, 3), 
    num_classes=5
)

# Split and train
split_idx = int(0.8 * n_samples)
X_rgb_train, X_rgb_test = X_rgb[:split_idx], X_rgb[split_idx:]
y_rgb_train, y_rgb_test = y_rgb[:split_idx], y_rgb[split_idx:]

rgb_model.fit(X_rgb_train, y_rgb_train)
rgb_pred = rgb_model.predict(X_rgb_test)
rgb_accuracy = accuracy(y_rgb_test, rgb_pred)

print(f"RGB Image Classification Accuracy: {rgb_accuracy:.4f}")
```

---

## 🎨 Generative Models

Generative models learn to create new data similar to the training data.

### Generative Adversarial Networks (GANs)

**Use Case**: Generate realistic synthetic data, especially images.

**Best For**:
- Image generation
- Data augmentation
- Creating synthetic datasets
- Style transfer
- Creative applications

**Example 1: Basic GAN Setup**

```python
from jackofalltrades.Models.GAN import GAN, Generator, Discriminator
import torch
import numpy as np
import matplotlib.pyplot as plt

# Initialize GAN components
print("Setting up GAN components...")

# Create individual components
generator = Generator(input_dim=100, output_channels=1)
discriminator = Discriminator(input_channels=1, feature_dim=64)

print(f"Generator input dimension: 100")
print(f"Generator output channels: 1 (grayscale)")
print(f"Discriminator feature dimension: 64")

# Create complete GAN
gan = GAN(noise_dim=100, image_channels=1)
print("GAN initialized successfully!")

# Generate random noise
batch_size = 16
noise = torch.randn(batch_size, 100)
print(f"Generated noise tensor shape: {noise.shape}")

# Generate images (before training)
with torch.no_grad():
    fake_images = gan.generator(noise)
    print(f"Generated images shape: {fake_images.shape}")

# Visualize generated images
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(16):
    row, col = i // 4, i % 4
    img = fake_images[i].squeeze().numpy()
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].axis('off')
plt.suptitle('GAN Generated Images (Untrained)')
plt.show()
```

**Example 2: GAN Training Setup**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create synthetic training data (simulating MNIST-like images)
def create_synthetic_dataset(n_samples=1000):
    """Create synthetic image dataset for GAN training"""
    images = []
    
    for i in range(n_samples):
        # Create simple patterns
        img = torch.zeros(1, 28, 28)
        
        # Random rectangles
        x1, y1 = torch.randint(0, 20, (2,))
        x2, y2 = x1 + torch.randint(5, 8, (1,)), y1 + torch.randint(5, 8, (1,))
        img[0, x1:x2, y1:y2] = torch.rand(1)
        
        # Random circles (approximated with filled squares)
        cx, cy = torch.randint(5, 23, (2,))
        r = torch.randint(2, 5, (1,))
        img[0, cx-r:cx+r, cy-r:cy+r] = torch.rand(1)
        
        images.append(img)
    
    return torch.stack(images)

# Create dataset
print("Creating synthetic dataset...")
real_images = create_synthetic_dataset(1000)
dataset = TensorDataset(real_images)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Batch size: 32")
print(f"Number of batches: {len(dataloader)}")

# Initialize GAN for training
gan = GAN(noise_dim=100, image_channels=1)

# Training would look like this (pseudo-code):
"""
# gan.train(dataloader, epochs=10)
"""

print("GAN training setup complete!")
print("Note: Actual training requires calling gan.train(dataloader, epochs)")
```

---

### Variational Autoencoders (VAEs)

**Use Case**: Generate new data and learn meaningful representations with JAX.

**Best For**:
- Dimensionality reduction
- Anomaly detection
- Data generation with control
- Learning latent representations
- Image compression

**Example 1: VAE Components Setup**

```python
from jackofalltrades.Models.VAE import EncoderDecoder
import jax.numpy as jnp
import numpy as np

# Initialize VAE model
print("Setting up Variational Autoencoder...")
vae_model = EncoderDecoder()
print("VAE EncoderDecoder initialized!")

# The VAE uses JAX for high-performance computation
print("VAE Framework: JAX (for high-performance ML)")
```

**Example 2: VAE Image Processing**

```python
from jackofalltrades.Models.VAE import (
    load_and_preprocess_image, 
    save_params, 
    load_params,
    optimizer
)
import numpy as np

# Image preprocessing example
print("VAE Image Processing Examples:")

# Note: These functions work with actual image files
# For demonstration, we'll show the API usage

print("1. Loading and preprocessing images:")
print("   img = load_and_preprocess_image('image.jpg', width=28, height=28)")

print("2. Saving trained model parameters:")
print("   save_params(params, state, 'vae_model.npz')")

print("3. Loading saved model parameters:")
print("   params, state = load_params('vae_model.npz')")

print("4. Setting up optimizer:")
print("   opt = optimizer(learning_rate=0.001)")

# Simulate latent space operations
latent_dim = 20
batch_size = 32

# Random latent vectors (what VAE would encode images to)
latent_vectors = np.random.normal(0, 1, (batch_size, latent_dim))
print(f"Latent space shape: {latent_vectors.shape}")
print(f"Latent dimension: {latent_dim}")

# VAE workflow explanation
print("\nVAE Workflow:")
print("1. Encoder: Image → Latent representation (mean, log_var)")
print("2. Sampling: Sample from latent distribution")
print("3. Decoder: Latent sample → Reconstructed image")
print("4. Loss: Reconstruction loss + KL divergence")
```

**Example 3: Complete VAE Training Conceptual Flow**

```python
# Conceptual VAE training flow (pseudo-code with actual API)
import jax
import jax.numpy as jnp

def vae_training_example():
    """
    Conceptual example of VAE training workflow
    """
    print("VAE Training Workflow Example:")
    
    # 1. Data preparation
    print("Step 1: Prepare image data")
    print("   - Load images with load_and_preprocess_image()")
    print("   - Normalize to [0, 1] range")
    print("   - Batch the data")
    
    # 2. Model initialization
    print("Step 2: Initialize VAE model")
    print("   - vae = EncoderDecoder()")
    print("   - Initialize parameters")
    
    # 3. Training loop
    print("Step 3: Training loop")
    print("   - Forward pass: encode → sample → decode")
    print("   - Compute loss: reconstruction + KL divergence")
    print("   - Backward pass: compute gradients")
    print("   - Update parameters")
    
    # 4. Model saving
    print("Step 4: Save trained model")
    print("   - save_params(params, state, 'trained_vae.npz')")
    
    # 5. Generation
    print("Step 5: Generate new images")
    print("   - Sample from latent space")
    print("   - Decode samples to images")
    
    return "VAE training workflow complete!"

result = vae_training_example()
print(f"\n{result}")
```

---

## 🔄 Model Selection Guide

### When to Use Each Model

| Task Type | Data Size | Complexity | Interpretability | Recommended Model |
|-----------|-----------|------------|------------------|-------------------|
| **Simple Regression** | Small-Medium | Low | High | Linear Regression |
| **Regression + Regularization** | Medium-Large | Medium | High | Ridge Regression |
| **Complex Patterns** | Large | High | Medium | Adaptive Regression |
| **Non-linear Regression** | Large | Very High | Low | MLP Regressor |
| **Binary Classification** | Any | Low-Medium | High | Logistic Regression |
| **Image Classification** | Large | High | Low | Image Classification |
| **Generate Images** | Very Large | Very High | Low | GAN |
| **Learn Representations** | Large | High | Medium | VAE |

### Performance Tips

1. **Data Preprocessing**: Always scale your features for neural networks
2. **Train/Test Split**: Use proper validation for model evaluation
3. **Overfitting**: Use regularization for complex models
4. **Data Size**: Start with simpler models for small datasets
5. **Feature Engineering**: Can improve simple model performance significantly

---

## 🚀 Next Steps

1. **Try the examples** - Run the code snippets with your own data
2. **Read the datasets guide** - Learn about available datasets
3. **Explore evaluation metrics** - Understand model performance measurement
4. **Check advanced examples** - See real-world applications

For more detailed information, see:
- [Datasets Reference](datasets_reference.md)
- [Evaluation Metrics Guide](evaluation_metrics.md)
- [API Documentation](api_reference.md) 