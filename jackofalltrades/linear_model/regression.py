import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Union
from jackofalltrades.Errors import r2score
from functools import partial
from sklearn.model_selection import train_test_split
import optax


class AdaptiveRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 5000,
        regularization_strength: float = 0.1,
        data_regularization=True,
        early_stop_patience=50,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = regularization_strength
        self.data_regularization = data_regularization
        self.early_stop_patience = early_stop_patience
        self.params = {}
        self.cost_history = []
        self.SS_X = StandardScaler() if data_regularization else None
        self.SS_y = StandardScaler() if data_regularization else None

    @staticmethod
    @jax.jit
    def _forward(X, params):
        """Vectorized forward pass supporting both single samples and batches."""
        return jnp.dot(X, params["w"]) + params["b"]

    @staticmethod
    @jax.jit
    def _single_forward(x, params):
        """Forward pass for single sample - optimized for vmap."""
        return jnp.dot(x, params["w"]) + params["b"]

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _loss(params, X, y, reg_strength):
        """Vectorized loss computation with L2 regularization."""
        y_pred = jnp.dot(X, params["w"]) + params["b"]
        mse = jnp.mean(jnp.square(y_pred - y))
        # L2 Regularization (Ridge) on weights only
        l2 = reg_strength * jnp.sum(jnp.square(params["w"]))
        return mse + l2

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _batch_loss(params, X_batch, y_batch, reg_strength):
        """Optimized batch loss computation using vmap."""
        # Vectorize loss computation across batch dimension
        # single_loss = lambda x, y: jnp.square(jnp.dot(x, params["w"]) + params["b"] - y)
        losses = jax.vmap(
            lambda x, y: jnp.square(jnp.dot(x, params["w"]) + params["b"] - y)
        )(X_batch, y_batch)
        mse = jnp.mean(losses)
        l2 = reg_strength * jnp.sum(jnp.square(params["w"]))
        return mse + l2

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        # 1. Data Scaling (Crucial for Gradient Descent)
        if self.data_regularization:
            X = self.SS_X.fit_transform(X)
            # Reshape y to (N, 1) and scale it
            y = self.SS_y.fit_transform(np.array(y).reshape(-1, 1))
        else:
            y = np.array(y).reshape(-1, 1)

        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32)

        # 2. Split for Validation (Internal split)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 3. Initialize Parameters
        n_features = X_train.shape[1]
        self.params = {
            "w": jnp.zeros((n_features, 1)),  # Start from 0
            "b": jnp.zeros((1,)),
        }

        # 4. Optimizer Setup
        optimizer = optax.adamw(self.learning_rate)
        opt_state = optimizer.init(self.params)

        # Define update step (JIT compiled)
        @jax.jit
        def update_step(params, opt_state, X_batch, y_batch):
            loss_val, grads = jax.value_and_grad(self._loss)(
                params, X_batch, y_batch, self.regularization_strength
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        # 5. Training Loop
        self.cost_history = []
        best_val_loss = float("inf")
        patience_counter = self.early_stop_patience

        description = tqdm(range(self.epochs))
        for i in description:
            # Train step
            self.params, opt_state, train_loss = update_step(
                self.params, opt_state, X_train, y_train
            )

            # Validation step (compute loss without grads)
            val_loss = self._loss(
                self.params, X_val, y_val, self.regularization_strength
            )

            self.cost_history.append(float(train_loss))

            # Update Progress Bar
            if i % 100 == 0:
                description.set_description(
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
                )

            # 6. Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = self.early_stop_patience  # Reset patience
            else:
                patience_counter -= 1

            if patience_counter <= 0:
                print(
                    f"\nEarly stopping at epoch {i}. Best Val Loss: {best_val_loss:.4f}"
                )
                break

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Optimized prediction using vectorized operations and vmap for large datasets."""
        if self.params is None:
            raise ValueError("Model not fitted yet.")

        if self.data_regularization:
            X = self.SS_X.transform(X)

        X = jnp.array(X, dtype=jnp.float32)

        # Use different strategies based on batch size
        if X.shape[0] <= 5000:
            # Direct vectorized computation for smaller batches
            y_pred = self._forward(X, self.params)
        else:
            # Use vmap for memory-efficient processing of large batches
            predict_single = jax.vmap(self._single_forward, in_axes=(0, None))
            y_pred = predict_single(X, self.params)

        # Inverse transform to get actual values back
        if self.data_regularization:
            return self.SS_y.inverse_transform(np.array(y_pred))

        return np.array(y_pred).flatten()

    def plot_cost(self):
        plt.plot(self.cost_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE + L2)")
        plt.title("Training Convergence")
        plt.show()

    def score(self, X, y):
        """Scikit-learn style score method (returns R^2)"""
        y_pred = self.predict(X)
        return r2score(y, y_pred)


class LogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        regularization_strength: float = 0.1,
        data_regularization=True,
        tol: float = 1e-4,
    ):
        """
        Scikit-learn style Logistic Regression using JAX.

        Parameters:
        - regularization_strength: Lambda parameter for L2 regularization.
          (Note: sklearn uses 'C' which is 1/lambda).
        - tol: Tolerance for stopping criteria (sklearn standard).
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = regularization_strength
        self.data_regularization = data_regularization
        self.tol = tol
        self.w = None
        self.b = None
        self.cost_history = []
        self.SS = StandardScaler() if data_regularization else None

    @staticmethod
    @jax.jit
    def _sigmoid(z):
        return 1 / (1 + jnp.exp(-z))

    @staticmethod
    def _loss(w, b, X, y, reg_strength):
        """
        Log Loss + L2 Regularization
        """
        m = X.shape[0]
        z = jnp.dot(X, w) + b
        y_pred = jax.nn.sigmoid(z)

        # Clip to prevent log(0) errors
        epsilon = 1e-15
        y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)

        # Standard Log Loss
        loss = -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))

        # Add L2 Regularization (Ridge)
        # Note: We do NOT regularize the bias 'b', only weights 'w'
        l2_cost = (reg_strength / (2 * m)) * jnp.sum(jnp.square(w))

        return loss + l2_cost

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> None:
        # 1. Prepare Data
        if self.data_regularization:
            X = self.SS.fit_transform(X)

        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32).reshape(-1, 1)

        m, n = X.shape

        # 2. Initialize Weights
        self.w = jnp.zeros((n, 1))
        self.b = jnp.zeros((1,))
        self.cost_history = []

        # 3. JIT Compile the Gradient Step for Speed
        @jax.jit
        def update_step(w, b, X, y):
            # value_and_grad returns (loss, (gradients))
            loss_val, (dw, db) = jax.value_and_grad(self._loss, argnums=(0, 1))(
                w, b, X, y, self.regularization_strength
            )
            return loss_val, dw, db

        # 4. Training Loop
        description = tqdm(range(self.epochs))
        for i in description:
            loss_val, dw, db = update_step(self.w, self.b, X, y)

            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            self.cost_history.append(float(loss_val))

            # Update Progress Bar
            if i % 100 == 0:
                description.set_description(f"Loss: {loss_val:.4f}")

            # 5. Sklearn-style Tolerance Stopping
            # Stop if loss isn't changing much anymore
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tol:
                print(f"Converged at epoch {i}")
                break

    def predict_proba(self, X):
        """Optimized probability prediction using vectorized operations."""
        if self.data_regularization:
            X = self.SS.transform(X)
        X = jnp.array(X, dtype=jnp.float32)

        # Use vmap for large batches to optimize memory usage
        if X.shape[0] > 5000:
            # single_proba = lambda x: self._sigmoid(jnp.dot(x, self.w) + self.b)
            vectorized_proba = jax.vmap(
                lambda x: self._sigmoid(jnp.dot(x, self.w) + self.b)
            )
            return vectorized_proba(X)
        else:
            # Direct vectorized computation for smaller batches
            return self._sigmoid(jnp.dot(X, self.w) + self.b)

    def predict(self, X) -> np.ndarray:
        """Optimized class label prediction."""
        proba = self.predict_proba(X)
        # Vectorized threshold operation
        return np.array((proba >= 0.5).astype(int)).flatten()

    @staticmethod
    @jax.jit
    def _batch_predict_proba(X, w, b):
        """JIT-compiled batch probability prediction."""
        return jax.nn.sigmoid(jnp.dot(X, w) + b)

    def plot_cost(self):
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.title("Training Cost")
        plt.show()


class RidgeRegression:
    def __init__(
        self,
        regularization_strength: float = 1.0,
        data_regularization=True,
        # Keeping these for compatibility as requested, though unused in Closed-Form
        learning_rate: float = None,
        epochs: int = None,
        early_stop_patience=None,
    ):
        self.regularization_strength = regularization_strength
        self.data_regularization = data_regularization
        self.params = None
        self.SS_X = StandardScaler() if data_regularization else None
        # We don't necessarily need to scale Y for closed-form,
        # but it helps interpretation if you want to add it back.
        # For strict sklearn logic, we usually just center data or fit intercept.

    def fit(self, X: Union[np.ndarray, jnp.ndarray], y: Union[np.ndarray, jnp.ndarray]):
        """
        Fits the Ridge Regression model using the Normal Equation (Closed-Form).
        Formula: w = (X^T X + alpha * I)^-1 X^T y
        """
        # 1. Data Scaling
        if self.data_regularization:
            X = self.SS_X.fit_transform(X)

        # Convert to JAX arrays
        X = jnp.array(X)
        y = jnp.array(y)

        # 2. Add Intercept (Bias) Term
        # We add a column of 1s to the left of X
        X_bias = jnp.hstack([jnp.ones((X.shape[0], 1)), X])

        # 3. Compute X Transpose X
        XTX = jnp.dot(X_bias.T, X_bias)

        # 4. Setup Regularization Matrix (Alpha * Identity)
        # We create an Identity matrix of size (n_features + 1)
        n_dims = XTX.shape[0]
        identity_matrix = jnp.eye(n_dims)

        # CRITICAL: Set the first diagonal element to 0.
        # We do NOT want to penalize (shrink) the Bias/Intercept term.
        identity_matrix = identity_matrix.at[0, 0].set(0)

        # 5. Compute X Transpose y
        XTy = jnp.dot(X_bias.T, y)

        # 6. Solve for Parameters (w)
        # (XTX + alpha * I) * w = XTy
        LHS = XTX + (self.regularization_strength * identity_matrix)

        self.params = jnp.linalg.solve(LHS, XTy)

        # print("Fit complete using Closed-Form Solution.")

    def predict(self, X: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
        """Optimized Ridge regression prediction with vmap support."""
        if self.params is None:
            raise ValueError("Model not fitted yet.")

        # 1. Scale Input
        if self.data_regularization:
            X = self.SS_X.transform(X)

        X = jnp.array(X)

        # 2. Optimized prediction based on batch size
        if X.shape[0] > 5000:
            # Use vmap for memory-efficient processing of large batches
            def single_predict(x):
                x_with_bias = jnp.concatenate([jnp.array([1.0]), x])
                return jnp.dot(x_with_bias, self.params)

            vectorized_predict = jax.vmap(single_predict)
            y_pred = vectorized_predict(X)
        else:
            # Direct vectorized computation for smaller batches
            X_bias = jnp.hstack([jnp.ones((X.shape[0], 1)), X])
            y_pred = jnp.dot(X_bias, self.params)

        return np.array(y_pred).flatten()


class LinearRegression:
    # self.params: jnp.ndarray

    def __init__(self, data_regularization=True):
        self.params = {}
        self.data_regularization = data_regularization

    def fit(self, X, y):
        if self.data_regularization:
            self.SS = StandardScaler()
            X = self.SS.fit_transform(X)
        X, y = jnp.array(X), jnp.array(y)
        X = jnp.hstack([jnp.ones((X.shape[0], 1)), X])
        XTX = jnp.dot(X.T, X)
        XTy = jnp.dot(X.T, y)
        self.params: np.ndarray = jnp.linalg.solve(XTX, XTy)

    def predict(self, X) -> np.ndarray:
        """Optimized Linear regression prediction with vmap support."""
        if self.data_regularization:
            X = self.SS.transform(X)
        X = jnp.array(X)

        # Optimized prediction based on batch size
        if X.shape[0] > 5000:
            # Use vmap for memory-efficient processing of large batches
            def single_predict(x):
                x_with_bias = jnp.concatenate([jnp.array([1.0]), x])
                return jnp.dot(x_with_bias, self.params)

            vectorized_predict = jax.vmap(single_predict)
            y_pred = vectorized_predict(X)
        else:
            # Direct vectorized computation for smaller batches
            X = jnp.hstack([jnp.ones((X.shape[0], 1)), X])
            y_pred = jnp.dot(X, self.params)

        return np.array(y_pred).flatten()

    @staticmethod
    def evaluate(y_true, y_pred) -> float:
        return r2score(y_true, y_pred)
