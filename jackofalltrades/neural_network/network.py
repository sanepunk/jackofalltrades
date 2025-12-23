from functools import partial
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from typing import Union
import sklearn.metrics as metrics
import jax
import optax
from jackofalltrades.Errors import r2score, accuracy
from flax import nnx
import optax


class MLPRegressor(nnx.Module):
    def __init__(self, learning_rate: float = 1e-4, epochs: int = 1000, regularization_strength: float = 1e-3, early_stop_patience=300, data_regularization = True, hidden_layers: int = 1, hidden_units: int = 10):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = regularization_strength
        self.early_stop_patience = early_stop_patience
        self.data_regularization = data_regularization
        self.hidden_layers = max(1, hidden_layers)
        self.hidden_units = hidden_units
        self.layers = []
        self.optimizer = None

    def _build_layers(self, input_dim: int):
        """Build MLP with correct input dimension (avoids dot shape errors)."""
        rngs = nnx.Rngs(42)
        layers = []
        # First layer maps input_dim -> hidden_units
        layers.append(nnx.Linear(input_dim, self.hidden_units, rngs=rngs))
        layers.append(nnx.relu)
        # Additional hidden layers (hidden_units -> hidden_units)
        for _ in range(self.hidden_layers - 1):
            layers.append(nnx.Linear(self.hidden_units, self.hidden_units, rngs=rngs))
            layers.append(nnx.relu)
        # Output layer
        layers.append(nnx.Linear(self.hidden_units, 1, rngs=rngs))
        self.layers = layers
        self.optimizer = nnx.Optimizer(self, optax.adam(self.learning_rate))

    def fit(self, X: Union[pd.DataFrame, np.ndarray, jnp.ndarray], y: Union[pd.Series, np.ndarray, jnp.ndarray]):
        try:
            self.train()
            if self.data_regularization:
                self.SS = StandardScaler()
                X = self.SS.fit_transform(X)
            X, y = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)
            # Build layers once input dimension is known
            if not self.layers:
                self._build_layers(X.shape[1])
            def loss(model, X, y):
                X = jnp.array(X, dtype=jnp.float32)
                for layer in model.layers:
                    X = layer(X)
                return jnp.mean(jnp.square(X - y))
            
            self.cost = []
            self.epoch = []
            description = tqdm(range(self.epochs))
            for i in description:
                loss_value = loss(self, X, y)
                grads = nnx.jit(nnx.grad(loss, argnums=(0)))(self, X, y)
                self.optimizer.update(grads)
                self.cost.append(loss_value)
                self.epoch.append(i)
                description.set_description(f"Cost: {self.cost[-1]}, Acc:")
                if loss_value >= self.cost[-1]:
                    self.early_stop_patience -= 1
                if self.early_stop_patience == 0:
                    print("Early Stopping")
                    break
        except Exception as e:
            print("An error occurred during fitting:", str(e))
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, jnp.ndarray]) -> np.ndarray:
        self.eval()
        if self.data_regularization:
            X = self.SS.transform(X)
        X = jnp.array(X, dtype=jnp.float32)
        # Ensure layers exist and match input dimensionality
        if not self.layers:
            self._build_layers(X.shape[1])
        for layer in self.layers:
            X = layer(X)
        return np.array(X)
