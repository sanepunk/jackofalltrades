import jax.numpy as jnp
import numpy as np
import jax
import pandas as pd

# from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from typing import Union
import optax
from flax import nnx


# class MLPRegressor(nnx.Module):
#     def __init__(
#         self,
#         learning_rate: float = 1e-4,
#         epochs: int = 1000,
#         regularization_strength: float = 1e-3,
#         early_stop_patience=300,
#         data_regularization=True,
#         hidden_layers: int = 1,
#         hidden_units: int = 10,
#     ):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.regularization_strength = regularization_strength
#         self.early_stop_patience = early_stop_patience
#         self.data_regularization = data_regularization
#         self.hidden_layers = max(1, hidden_layers)
#         self.hidden_units = hidden_units
#         self.layers = []
#         self.optimizer = None

#     def _build_layers(self, input_dim: int):
#         """Build MLP with modern nnx architecture for optimal vectorization."""
#         rngs = nnx.Rngs(42)
#         layers = []

#         # First hidden layer
#         layers.append(nnx.Linear(input_dim, self.hidden_units, rngs=rngs))

#         # Additional hidden layers
#         for _ in range(self.hidden_layers - 1):
#             layers.append(nnx.Linear(self.hidden_units, self.hidden_units, rngs=rngs))

#         # Output layer
#         layers.append(nnx.Linear(self.hidden_units, 1, rngs=rngs))

#         self.layers = layers
#         self.optimizer = nnx.Optimizer(self, optax.adam(self.learning_rate))

#     def __call__(self, x):
#         """Vectorized forward pass through the network."""
#         # Efficiently process batches using JAX operations
#         for i, layer in enumerate(self.layers[:-1]):  # All but last layer
#             x = layer(x)
#             x = nnx.relu(x)  # Apply activation

#         # Final layer (no activation for regression)
#         x = self.layers[-1](x)
#         return x

#     def fit(
#         self,
#         X: Union[pd.DataFrame, np.ndarray, jnp.ndarray],
#         y: Union[pd.Series, np.ndarray, jnp.ndarray],
#     ):
#         try:
#             self.train()
#             if self.data_regularization:
#                 self.SS = StandardScaler()
#                 X = self.SS.fit_transform(X)
#             X, y = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

#             # Ensure y is properly shaped
#             if y.ndim == 1:
#                 y = y.reshape(-1, 1)

#             # Build layers once input dimension is known
#             if not self.layers:
#                 self._build_layers(X.shape[1])

#             # Vectorized loss function using the __call__ method
#             def loss_fn(model, X_batch, y_batch):
#                 predictions = model(X_batch)
#                 mse_loss = jnp.mean(jnp.square(predictions - y_batch))
#                 # Add L2 regularization
#                 l2_loss = 0.0
#                 for layer in model.layers:
#                     if hasattr(layer, "kernel"):
#                         l2_loss += jnp.sum(layer.kernel**2)
#                 return mse_loss + self.regularization_strength * l2_loss

#             self.cost = []
#             self.epoch = []
#             best_loss = float("inf")
#             patience_counter = self.early_stop_patience

#             # JIT compile the training step for maximum performance
#             @nnx.jit
#             def train_step(model, optimizer, X_batch, y_batch):
#                 loss_val, grads = nnx.value_and_grad(loss_fn)(model, X_batch, y_batch)
#                 optimizer.update(grads)
#                 return loss_val

#             description = tqdm(range(self.epochs))
#             for i in description:
#                 loss_value = train_step(self, self.optimizer, X, y)

#                 self.cost.append(float(loss_value))
#                 self.epoch.append(i)

#                 # Early stopping logic
#                 if loss_value < best_loss:
#                     best_loss = loss_value
#                     patience_counter = self.early_stop_patience
#                 else:
#                     patience_counter -= 1

#                 description.set_description(
#                     f"Loss: {loss_value:.6f} | Best: {best_loss:.6f}"
#                 )

#                 if patience_counter <= 0:
#                     print(f"\nEarly stopping at epoch {i}")
#                     break

#         except Exception as e:
#             print("An error occurred during fitting:", str(e))

#     def predict(self, X: Union[pd.DataFrame, np.ndarray, jnp.ndarray]) -> np.ndarray:
#         """Vectorized prediction using efficient batch processing."""
#         self.eval()

#         if self.data_regularization:
#             X = self.SS.transform(X)
#         X = jnp.array(X, dtype=jnp.float32)

#         # Ensure layers exist and match input dimensionality
#         if not self.layers:
#             self._build_layers(X.shape[1])

#         # For small batches, use direct forward pass
#         if X.shape[0] <= 1000:
#             predictions = self(X)
#         else:
#             # For large batches, use vmap for memory efficiency
#             # Process single examples and vmap across batch dimension
#             # single_predict = lambda x: self(x.reshape(1, -1)).squeeze()
#             vectorized_predict = nnx.vmap(
#                 lambda x: self(x.reshape(1, -1)).squeeze(), in_axes=0
#             )
#             predictions = vectorized_predict(X).reshape(-1, 1)

#         return np.array(predictions).flatten()


class MLPRegressor(nnx.Module):
    """
    JAX/Flax-nnx MLP regressor.

    Training is a single `jax.lax.fori_loop` with a fixed length of `self.epochs`
    - no Python `for`/`while` loop, no `tqdm`, no `print`/stdio anywhere inside
    `fit`. Every host<->device sync inside a loop iteration (a `float()` call,
    an `if loss < x:` branch, a progress-bar update) forces the device to
    finish and blocks the next dispatch, which serializes what should be a
    tight, fully-compiled loop. Removing all of that is the actual win here;
    it isn't just cosmetic.

    Consequence: `jax.lax.fori_loop` requires a fixed number of iterations
    known ahead of time - it cannot branch out of the loop early the way a
    Python `while` with a `break` can. Early stopping is therefore implemented
    as an in-loop *freeze* instead of an early *exit*: once the patience
    counter (tracked as a device-side value, not a Python int) reaches zero,
    the parameter/optimizer update for that step is discarded via
    `jnp.where(...)` and the state carried through unchanged for the rest of
    the fixed-length loop. The loop always runs all `self.epochs` iterations
    on-device, but training effectively stops the moment patience runs out -
    you just don't get the wall-clock saving of literally exiting early, since
    that would require a host-visible branch. The full per-epoch loss curve
    is still recorded (in a preallocated on-device buffer) and pulled to host
    exactly once, after the loop finishes - the only host sync in `fit`.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        epochs: int = 1000,
        regularization_strength: float = 1e-3,
        early_stop_patience: int = 300,
        data_regularization: bool = True,
        hidden_layers: int = 1,
        hidden_units: int = 10,
    ):
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
        self.SS = None
        self.cost = []
        self.epoch = []

    def _build_layers(self, input_dim: int):
        """Build MLP with modern nnx architecture for optimal vectorization."""
        rngs = nnx.Rngs(42)
        layers = [nnx.Linear(input_dim, self.hidden_units, rngs=rngs)]
        for _ in range(self.hidden_layers - 1):
            layers.append(nnx.Linear(self.hidden_units, self.hidden_units, rngs=rngs))
        layers.append(nnx.Linear(self.hidden_units, 1, rngs=rngs))

        self.layers = layers
        self.optimizer = nnx.Optimizer(self, optax.adam(self.learning_rate))

    def __call__(self, x):
        """Vectorized forward pass through the network."""
        for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))
        return self.layers[-1](x)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, jnp.ndarray],
        y: Union[pd.Series, np.ndarray, jnp.ndarray],
    ):
        self.train()
        if self.data_regularization:
            self.SS = StandardScaler()
            X = self.SS.fit_transform(X)
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if not self.layers:
            self._build_layers(X.shape[1])

        reg_strength = self.regularization_strength
        patience_full = jnp.asarray(self.early_stop_patience, dtype=jnp.int32)

        def loss_fn(model, X_batch, y_batch):
            predictions = model(X_batch)
            mse_loss = jnp.mean(jnp.square(predictions - y_batch))
            l2_loss = 0.0
            for layer in model.layers:
                if hasattr(layer, "kernel"):
                    l2_loss += jnp.sum(layer.kernel**2)
            return mse_loss + reg_strength * l2_loss

        graphdef, state = nnx.split((self, self.optimizer))

        def step(state):
            model, optimizer = nnx.merge(graphdef, state)
            loss_val, grads = nnx.value_and_grad(loss_fn)(model, X, y)
            optimizer.update(grads)
            new_state = nnx.state((model, optimizer))
            return new_state, loss_val

        def body_fn(i, carry):
            state, best_loss, patience, losses_buf = carry
            new_state, loss_val = step(state)

            still_training = patience > 0
            improved = loss_val < best_loss

            # Freeze the update once patience is exhausted, instead of exiting
            # the (fixed-length) loop.
            committed_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(still_training, new, old),
                new_state,
                state,
            )
            committed_loss = jnp.where(still_training, loss_val, best_loss)

            new_best = jnp.where(still_training & improved, loss_val, best_loss)
            new_patience = jnp.where(
                still_training,
                jnp.where(improved, patience_full, patience - 1),
                patience,
            )

            losses_buf = jax.lax.dynamic_update_slice_in_dim(
                losses_buf, committed_loss[None], i, axis=0
            )
            return committed_state, new_best, new_patience, losses_buf

        init_carry = (
            state,
            jnp.asarray(jnp.inf, dtype=jnp.float32),
            patience_full,
            jnp.zeros((self.epochs,), dtype=jnp.float32),
        )

        final_state, _, _, losses_buf = jax.lax.fori_loop(
            0, self.epochs, body_fn, init_carry
        )

        model, optimizer = nnx.merge(graphdef, final_state)
        nnx.update(self, nnx.state(model))
        self.optimizer = optimizer

        # Single host sync for the whole run, purely for reporting/inspection.
        losses_host = np.asarray(losses_buf)
        self.cost = losses_host.tolist()
        self.epoch = list(range(self.epochs))

    def predict(self, X: Union[pd.DataFrame, np.ndarray, jnp.ndarray]) -> np.ndarray:
        """Vectorized prediction using a plain batched forward pass."""
        self.eval()

        if self.data_regularization:
            X = self.SS.transform(X)
        X = jnp.asarray(X, dtype=jnp.float32)

        if not self.layers:
            self._build_layers(X.shape[1])

        @nnx.jit
        def forward(model, X_batch):
            return model(X_batch)

        n = X.shape[0]
        chunk_size = 4096
        if n <= chunk_size:
            return np.array(forward(self, X)).flatten()

        n_chunks = -(-n // chunk_size)
        padded_n = n_chunks * chunk_size
        pad = padded_n - n
        X_padded = jnp.pad(X, ((0, pad), (0, 0))) if pad > 0 else X
        preds_buffer = jnp.zeros((padded_n, 1), dtype=jnp.float32)

        def body_fn(idx, buf):
            start = idx * chunk_size
            chunk = jax.lax.dynamic_slice_in_dim(X_padded, start, chunk_size, axis=0)
            chunk_preds = forward(self, chunk)
            return jax.lax.dynamic_update_slice_in_dim(buf, chunk_preds, start, axis=0)

        preds_buffer = jax.lax.fori_loop(0, n_chunks, body_fn, preds_buffer)
        return np.array(preds_buffer[:n]).flatten()
