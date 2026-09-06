# 🎯 jackofalltrades

[![PyPI version](https://badge.fury.io/py/jackofalltrades.svg)](https://badge.fury.io/py/jackofalltrades)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> *A jack of all trades is a master of none, but oftentimes better than a master of one.*

## 🌟 Overview

`jackofalltrades` is a comprehensive, beginner-friendly machine learning library designed to make ML accessible to everyone. Whether you're a student learning the fundamentals, a researcher prototyping new ideas, or a developer integrating ML into your applications, jackofalltrades provides the tools you need.

### ✨ Key Features

- **🚀 One-line dataset loading** - Access popular datasets instantly
- **🧠 Complete ML pipeline** - From data loading to model evaluation
- **🔧 Easy-to-use APIs** - Intuitive interfaces for all skill levels
- **📊 Built-in evaluation** - Comprehensive metrics and error analysis
- **🎨 Advanced models** - GANs, VAEs, and deep learning models
- **💡 Educational focus** - Perfect for learning and teaching ML concepts

### 🎯 What Can You Do?

- **Regression Tasks**: Linear, Ridge, Adaptive, and MLP regression models
- **Classification**: Logistic regression and image classification
- **Generative Models**: GANs and Variational Autoencoders
- **Data Analysis**: Load and explore real-world datasets
- **Model Evaluation**: Comprehensive metrics and performance analysis

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install from PyPI

```bash
pip install jackofalltrades
```

### Install from Source

```bash
git clone https://github.com/sanepunk/jackofalltrades.git
cd jackofalltrades
pip install -e .
```

### Development Installation

For developers who want to contribute:

```bash
git clone https://github.com/sanepunk/jackofalltrades.git
cd jackofalltrades
pip install -e ".[dev]"
```

## ⚡ Quick Start

### 🏠 Real Estate Price Prediction

```python
from jackofalltrades.datasets import get_real_estate
from jackofalltrades.Models import LinearRegression
from jackofalltrades.Errors import Error

# Load dataset
X, y = get_real_estate()

# Train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate performance
evaluator = Error(y_true=y, y_predicted=predictions)
print(f"R² Score: {evaluator.RSquared():.3f}")
print(f"MSE: {evaluator.MSE():.3f}")
```

### 🖼️ Image Classification

```python
from jackofalltrades.Models import ImageClassification
import numpy as np

# Create sample data (28x28 grayscale images)
X = np.random.rand(1000, 28, 28, 1)
y = np.random.randint(0, 10, 1000)

# Initialize and train model
model = ImageClassification(input_shape=(28, 28, 1), num_classes=10)
model.fit(X, y)

# Make predictions
predictions = model.predict(X[:10])
```

### 🎲 Generate Images with GANs

```python
from jackofalltrades.Models.GAN import GAN
import torch

# Create GAN
gan = GAN(noise_dim=100, image_channels=1)

# Train (assuming you have a data_loader)
# gan.train(data_loader, epochs=10)

# Generate images
noise = torch.randn(16, 100)  # Generate 16 images
generated_images = gan.generator(noise)
```

## 📚 Documentation Structure

This package includes comprehensive documentation:

- **[Models Guide](models_guide.md)** - Detailed guide to all available models
- **[Datasets Reference](datasets_reference.md)** - Complete dataset documentation
- **[Evaluation Metrics](evaluation_metrics.md)** - Guide to error analysis and metrics
- **[Examples](examples/)** - Real-world usage examples

## 🏗️ Package Architecture

```
jackofalltrades/
├── datasets.py          # Dataset loading utilities
├── Errors.py           # Evaluation metrics and error analysis
├── Models/             # Core ML models
│   ├── __init__.py
│   ├── GAN/           # Generative Adversarial Networks
│   └── VAE/           # Variational Autoencoders
└── utils/             # Utility functions
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by scikit-learn's ease of use
- Built for the machine learning community
- Thanks to all contributors and users

## 📞 Support

- 📧 Email: [patil.ojas.250202@gmail.com](mailto:patil.ojas.250202@gmail.com)
- 🐛 Issues: [GitHub Issues](https://github.com/sanepunk/jackofalltrades/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/sanepunk/jackofalltrades/discussions)

---

*Made with ❤️ for the machine learning community*
