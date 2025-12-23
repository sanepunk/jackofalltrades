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


# class AdaptiveRegression:
    # def __init__(self, learning_rate: float = 0.03, epochs: int = 10000, regularization_strength: float = 0.1,
    #              data_regularization=True) -> None:
    #     """
    #     Initialize the LinearRegression object.

    #     Parameters:
    #     - learning_rate: Learning rate for gradient descent (default = 0.03).
    #     - epochs: Number of training iterations (default = 10000).
    #     - regularization_strength (default = 0.1)
    #     """
    #     self.n = None
    #     self.m = None
    #     try:
    #         self.cost = []
    #         self.epoch = []
    #         self.data_regularization = data_regularization
    #         self.regularization_strength = regularization_strength
    #         self.learning_rate = learning_rate
    #         self.epochs = epochs
    #         self.w = None
    #         self.b = None
    #         self.params = {}
    #     except Exception as e:
    #         print("An error occurred during initialization:", str(e))

    # @partial(jax.jit, static_argnums=0)
    # def forward(self, X, params):
    #     return jnp.dot(X, params['w']) + params['b']

    # @partial(jax.jit, static_argnums=0)
    # def loss(self, params, X, y):
    #     y_pred = self.forward(X, params)
    #     return jnp.mean(jnp.square(y_pred - y))

    # @partial(jax.jit, static_argnums=0)
    # def update(self, params, grads, opt_state):
    #     updates, opt_state = optax.adamw(self.learning_rate).update(grads, opt_state, params)
    #     new_params = optax.apply_updates(params, updates)
    #     return new_params, opt_state

    # def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], validation_split=0.2,
    #         early_stop_patience=5) -> None:
    #     """
    #     Train the linear regression model using gradient descent.
    #     Parameters:
    #     - X: Input features as a pandas DataFrame.
    #     - y: Target variable as a pandas Series.
    #     """
    #     try:
    #         X, y = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)
    #         if self.data_regularization:
    #             self.SS = StandardScaler()
    #             X = self.SS.fit_transform(X)
    #         self.m, self.n = X.shape
    #         model = LinearRegression()
    #         model.fit(X, y)
    #         self.w = jnp.array(model.params[:-1])
    #         self.b = jnp.array(model.params[-1])
    #         self.params = {'w': self.w, 'b': self.b}
    #         description = tqdm(range(self.epochs))
    #         X, X_test, y, y_test = train_test_split(X, y, test_size=validation_split, random_state=42)
    #         best_val_loss = float('inf')
    #         best_val_acc = float('-inf')
    #         patience = early_stop_patience
    #         solver = optax.adamw(learning_rate=0.003)
    #         opt_state = solver.init(self.params)
    #         loss, grads = jax.value_and_grad(self.loss, argnums=0, allow_int=True)(self.params, X, y)
    #         # acc = round(r2score(y, self.forward(X, self.params)), 5)
    #         self.params, opt_state = self.update(self.params, grads, opt_state)
    #         if metrics.r2_score(y, np.array(self.forward(X, self.params))) < 0:
    #             tqdm.write("Negative R2Score, wait for a while")
    #         for i in description:
    #             acc = 0
    #             loss = 0
    #             description.set_description(f"R2Score:{r2score(y, self.forward(X, self.params))}")
    #             for _ in range(10):
    #                 loss, grads = jax.value_and_grad(self.loss, argnums=0, allow_int=True)(self.params, X, y)
    #                 acc = round(r2score(y, self.forward(X, self.params)), 5)
    #                 self.params, opt_state = self.update(self.params, grads, opt_state)
    #                 # self.params = self.update(self.params, grads)
    #                 self.cost.append(loss)
    #                 self.epoch.append(i)

    #             if acc <= best_val_acc or loss >= best_val_loss:
    #                 patience -= 1
    #             else:
    #                 best_val_loss = loss
    #                 best_val_acc = acc
    #                 patience = early_stop_patience

    #             if patience == 0:
    #                 tqdm.write(f"Stopping early at epoch {i+1} due to constant or slow convergence rate")
    #                 if r2score(y, self.forward(X, self.params)) < .5:
    #                     print('Try changing the hyperparameters')
    #                 description.close()
    #                 break
    #         if r2score(y, self.forward(X, self.params)) <= .5:
    #             print("Model isn't working well try: ")
    #             print("1. Changing the Hyperparameters")
    #             print("2. Changing the Model e.x., MLPRegressor")

    #     except Exception as e:
    #         print("An error occurred during fitting:", str(e))

    # def predict(self, X_test: pd.DataFrame) -> np.ndarray:
    #     """
    #     Predict the target variable for the given input features.

    #     Parameters:
    #     - X_test: Input features for prediction as a pandas DataFrame.

    #     Returns:
    #     - Predicted target variable as a numpy array.
    #     """
    #     if self.data_regularization:
    #         X_test = self.SS.transform(X_test)
    #     return np.array(self.forward(X_test, self.params))

    # def plot_cost(self) -> None:
    #     """
    #     Plot the cost function over training iterations.
    #     """
    #     plt.plot(self.cost, self.epoch)
    #     plt.show()

    # @staticmethod
    # def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    #     """
    #     Evaluate the model using the R-squared metric.

    #     Parameters:
    #     - X_test: Test input features as a numpy array.
    #     - y_test: Test target variable as a numpy array.
    #     """
    #     return r2score(y_true, y_pred)

class AdaptiveRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 5000, 
                 regularization_strength: float = 0.1, data_regularization=True,
                 early_stop_patience=50):
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
    @jax.jit # JIT compile, keeping 'reg_strength' static if needed
    def _forward(X, params):
        return jnp.dot(X, params['w']) + params['b']

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _loss(params, X, y, reg_strength):
        y_pred = jnp.dot(X, params['w']) + params['b']
        mse = jnp.mean(jnp.square(y_pred - y))
        # L2 Regularization (Ridge) on weights only
        l2 = reg_strength * jnp.sum(jnp.square(params['w']))
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Initialize Parameters
        n_features = X_train.shape[1]
        self.params = {
            'w': jnp.zeros((n_features, 1)), # Start from 0
            'b': jnp.zeros((1,))
        }

        # 4. Optimizer Setup
        optimizer = optax.adamw(self.learning_rate)
        opt_state = optimizer.init(self.params)

        # Define update step (JIT compiled)
        @jax.jit
        def update_step(params, opt_state, X_batch, y_batch):
            loss_val, grads = jax.value_and_grad(self._loss)(params, X_batch, y_batch, self.regularization_strength)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        # 5. Training Loop
        self.cost_history = []
        best_val_loss = float('inf')
        patience_counter = self.early_stop_patience
        
        description = tqdm(range(self.epochs))
        for i in description:
            # Train step
            self.params, opt_state, train_loss = update_step(self.params, opt_state, X_train, y_train)
            
            # Validation step (compute loss without grads)
            val_loss = self._loss(self.params, X_val, y_val, self.regularization_strength)
            
            self.cost_history.append(float(train_loss))

            # Update Progress Bar
            if i % 100 == 0:
                description.set_description(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # 6. Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = self.early_stop_patience # Reset patience
            else:
                patience_counter -= 1
            
            if patience_counter <= 0:
                print(f"\nEarly stopping at epoch {i}. Best Val Loss: {best_val_loss:.4f}")
                break

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.params is None:
            raise ValueError("Model not fitted yet.")
            
        if self.data_regularization:
            X = self.SS_X.transform(X)
            
        X = jnp.array(X, dtype=jnp.float32)
        y_pred = self._forward(X, self.params)
        
        # Inverse transform to get actual values back
        if self.data_regularization:
            return self.SS_y.inverse_transform(np.array(y_pred))
        
        return np.array(y_pred)

    def plot_cost(self):
        plt.plot(self.cost_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE + L2)')
        plt.title('Training Convergence')
        plt.show()

    def score(self, X, y):
        """Scikit-learn style score method (returns R^2)"""
        y_pred = self.predict(X)
        return r2score(y, y_pred)


# class LogisticRegression:

#     def __init__(self, learning_rate: float = 0.03, epochs: int = 10000, regularization_strength: float = 0.1,
#                  data_regularization=True) -> None:
#         """
#         Initialize the LogisticRegression object.

#         Parameters:
#         - learning_rate: Learning rate for gradient descent (default = 0.03).
#         - epochs: Number of training iterations (default = 10000).
#         - regularization_strength (default = 0.1)
#         """
#         try:
#             self.data_regularization = data_regularization
#             self.regularization_strength = regularization_strength
#             self.learning_rate = learning_rate
#             self.epochs = epochs
#         except Exception as e:
#             print("An error occurred during initialization:", str(e))

#     @staticmethod
#     def loss(w, b, X, y) -> jnp.ndarray:
#         y_pred = jax.nn.sigmoid(jnp.dot(X, w) + b)
#         return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))

#     def fit(self, X: Union[pd.DataFrame, np.ndarray, jnp.ndarray], y: Union[pd.Series, np.ndarray, jnp.ndarray]) -> None:
#         """
#         Train the logistic regression model using gradient descent.
#         Parameters:
#         - X: Input features as a pandas DataFrame.
#         - y: Target variable as a pandas Series.
#         """
#         try:
#             X, y = jnp.array(X, dtype=np.float32), jnp.array(y, dtype=np.float32).reshape(-1, 1)
#             if self.data_regularization:
#                 self.SS = StandardScaler()
#                 X = self.SS.fit_transform(X)
#             X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                                                     test_size=0.01, random_state=42)
#             self.m, self.n = X_train.shape
#             self.w = jnp.zeros((self.n, 1))
#             self.b = jnp.zeros((1,))
#             self.cost = []
#             self.epoch = []
#             description = tqdm(range(self.epochs))
#             for i in description:
#                 loss_value = self.loss(self.w, self.b, X_train, y_train)
#                 grads = jax.grad(self.loss, argnums=(0, 1))(self.w, self.b, X_train, y_train)
#                 dw, db = grads
#                 self.w -= self.learning_rate * dw
#                 self.b -= self.learning_rate * db
#                 if loss_value >= self.cost[-1]:
#                     print("Early Stopping")
#                     break
#                 self.cost.append(loss_value)
#                 self.epoch.append(i)
#                 description.set_description(f"Cost: {self.cost[-1]}, Acc: {accuracy(y_train, jax.nn.sigmoid(jnp.dot(X_train, self.w) + self.b) > 0.5)}")

#         except Exception as e:
#             print("An error occurred during fitting:", str(e))

#     def predict(self, X_test: Union[pd.DataFrame, np.ndarray, jnp.ndarray]) -> np.ndarray:
#         """
#         Predict the target variable for the given input features.

#         Parameters:
#         - X_test: Input features for prediction as a pandas DataFrame.

#         Returns:
#         - Predicted target variable as a numpy array.
#         """
#         if self.data_regularization:
#             X_test = self.SS.transform(X_test)
#         return np.array(jax.nn.sigmoid(jnp.dot(np.array(X_test, dtype=jnp.float32)) > 0.5).astype(int))

#     def plot_cost(self) -> None:
#         """
#         Plot the cost function over training iterations.
#         """
#         plt.plot(self.epoch, self.cost)
#         plt.xlabel('Epoch')
#         plt.ylabel('Cost')
#         plt.title('Cost vs Epochs')
#         plt.show()

#     @staticmethod
#     def evaluate(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
#         """
#         Evaluate the model using the accuracy metric.

#         Parameters:
#         - y_true: True labels as a numpy array.
#         - y_predicted: Predicted labels as a numpy array.
#         """

#         return accuracy(y_true, y_predicted)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Union

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, 
                 regularization_strength: float = 0.1, data_regularization=True,
                 tol: float = 1e-4):
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

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
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
            loss_val, (dw, db) = jax.value_and_grad(self._loss, argnums=(0, 1))(w, b, X, y, self.regularization_strength)
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
        """Returns probability (0 to 1)"""
        if self.data_regularization:
            X = self.SS.transform(X)
        X = jnp.array(X, dtype=jnp.float32)
        return self._sigmoid(jnp.dot(X, self.w) + self.b)

    def predict(self, X) -> np.ndarray:
        """Returns class labels (0 or 1)"""
        proba = self.predict_proba(X)
        return np.array((proba >= 0.5).astype(int))

    def plot_cost(self):
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('Training Cost')
        plt.show()


# class MLPRegressor(tf.keras.Model):
#     def __init__(self, learning_rate: float = 1.4e-4, epochs: int = 10000, regularization_strength: float = 0.1,
#                  data_regularization=True, hidden_layers: int = 1) -> None:
#         """
#         Initialize the MLPRegressor object.

#         Parameters:
#         - learning_rate: Learning rate for gradient descent (default = 0.03).
#         - epochs: Number of training iterations (default = 10000).
#         - regularization_strength (default = 0.1)
#         - data_regularization: Whether to standardize the input data (default = True).
#         - hidden_layers: Number of hidden layers in the neural network (default = 1).
#         - hidden_units: Number of units in each hidden layer (default = 10 per layer).
#         - Architecture: [input_units, {hidden_units}, output_unit = 1]
#         """

#         try:
#             self.y = None
#             self.data_regularization = data_regularization
#             self.regularization_strength = regularization_strength
#             self.learning_rate = learning_rate
#             self.epochs = epochs
#             self.hidden_layers = hidden_layers
#             self.X = None
#             self.cache_model: tf.keras.Model = None

#         except Exception as e:
#             print("An error occurred during initialization:", str(e))

#     def fit(self, X: Union[pd.DataFrame, np.ndarray, tf.Tensor], y: Union[pd.DataFrame, np.ndarray, tf.Tensor]) -> None:
#         """
#         Train the linear regression model using gradient descent.
#         Parameters:
#         - X: Input features as a pandas DataFrame or numpy array or tensorflow Tensor.
#         - y: Target variable as a pandas Series or numpy array or tensorflow Tensor.
#         """
#         try:
#             X = np.array(X, dtype=np.float32)
#             y = np.array(y, dtype=np.float32)
#             if self.data_regularization:
#                 self.SS = StandardScaler()
#                 X = self.SS.fit_transform(X)
#             model = tf.keras.models.Sequential()
#             model.add(tf.keras.layers.InputSpec(shape=(X.shape[1],)))
#             for i in range(self.hidden_layers):
#                 model.add(tf.keras.layers.Dense(10, activation='relu',
#                                                 activity_regularizer=tf.keras.regularizers.l2(
#                                                     l2=self.regularization_strength)))
#                 model.add(tf.keras.layers.Dropout(0.2))
#             model.add(tf.keras.layers.Dense(1))
#             model.compile(optimizer=tf.keras.optimizers.legacy.SGD(self.learning_rate),
#                           loss=tf.keras.losses.MeanSquaredLogarithmicError(reduction='auto',
#                                                                            name='mean_squared_logarithmic_error'))
#             model.fit(X, y, epochs=self.epochs, verbose=1, batch_size=20)
#             self.cache_model = model
#         except Exception as e:
#             print("An error occurred during fitting:", str(e))

#     def predict(self, X_test: Union[pd.DataFrame, tf.Tensor, np.ndarray]) -> Union[np.ndarray]:
#         """
#         Predict the target variable for the given input features.

#         Parameters:
#         - X_test: Input features for prediction as a pandas DataFrame.

#         Returns:
#         - Predicted target variable as a numpy array.
#         """
#         X_test = np.array(X_test, dtype=np.float32)
#         if self.data_regularization:
#             X_test = self.SS.fit_transform(X_test)
#         return np.array(self.cache_model.predict(X_test))

#     @staticmethod
#     def evaluate(y_true: Union[pd.Series, tf.Tensor, np.ndarray],
#                  y_pred: Union[pd.Series, tf.Tensor, np.ndarray]) -> float:
#         """
#         Evaluate the model using the R-squared metric.

#         Parameters:
#         - y_true: Test input features as a numpy array or pandas Series or tensorflow torch.
#         - y_test: Test target variable as a numpy array.
#         """
#         return r2score(y_true, y_pred)

#     def save_model(self, path: str) -> None:
#         """
#         Save the trained model to a file.

#         Parameters:
#         - path: Path to save the model file.
#         """
#         super().save_model(path)

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

# class RidgeRegression:
#     def __init__(self, learning_rate: float = 1e-4, epochs: int = 1000, regularization_strength: float = 1e-3, early_stop_patience=300, data_regularization = True):
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.regularization_strength = regularization_strength
#         self.w = None
#         self.b = None
#         self.params = {}
#         self.cost = []
#         self.epoch = []
#         self.early_stop_patience = early_stop_patience
#         self.data_regularization = data_regularization

#     @partial(jax.jit, static_argnums=0)
#     def forward(self, X, params):
#         return jnp.dot(X, params['w']) + params['b']

#     @partial(jax.jit, static_argnums=0)
#     def loss(self, params, X, y) -> jnp.ndarray:
#         y_pred = self.forward(X, params)
#         mse_loss = jnp.mean(jnp.square(y_pred - y))
#         l2_regularization = self.regularization_strength * jnp.mean(jnp.square(params['w']))
#         return mse_loss + l2_regularization
        
#     @partial(jax.jit, static_argnums=0)
#     def update(self, params, opt_state, grads):
#         new_params, opt_state = optax.adam(self.learning_rate).update(grads, opt_state, params)
#         return new_params, opt_state
    
#     def fit(self, X, y):
#         if self.data_regularization:
#             self.SS = StandardScaler()
#             X = self.SS.fit_transform(X)
#         X, y = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)
#         self.w = jnp.array(np.random.normal(size=X.shape[1]) * 1e-4)
#         self.b = jnp.zeros((1,))
#         self.params = {'w': self.w, 'b': self.b}
#         opt_state = optax.adam(self.learning_rate).init(self.params)
#         best_val_acc, best_val_loss = float('-inf'), float('inf')
#         acc = 0
#         loss = 0
#         description = tqdm(range(self.epochs))
#         patience = self.early_stop_patience
#         for i in description:
#             if i % 100 == 0:
#                 print(f"Epoch {i} - Loss: {self.loss(self.params, X, y)}")
#             loss_value = self.loss(self.params, X, y)
#             grads = jax.grad(self.loss, argnums=0)(self.params, X, y)
#             self.params, opt_state = self.update(self.params, opt_state, grads)
#             self.cost.append(loss_value)
#             self.epoch.append(i)
#             if acc <= best_val_acc or loss >= best_val_loss:
#                     patience -= 1
#             else:
#                     best_val_loss = loss
#                     best_val_acc = acc
#                     patience = self.early_stop_patience

#             if patience == 0:
#                     tqdm.write(f"Stopping early at epoch {i+1} due to constant or slow convergence rate")
#                     if r2score(y, self.forward(X, self.params)) < .5:
#                         print('Try changing the hyperparameters')
#                     description.close()
#                     break
#         if r2score(y, self.forward(X, self.params)) <= .5:
#             print("Model isn't working well try: ")
#             print("1. Changing the Hyperparameters")
#             print("2. Changing the Model e.x., MLPRegressor")

#     def predict(self, X_test) -> np.ndarray:
#         if self.data_regularization:
#             X_test = self.SS.transform(X_test)
#         return np.array(self.forward(X_test, self.params))

#     @staticmethod
#     def evaluate(y_true, y_pred) -> float:
#         return r2score(y_true, y_pred)

class RidgeRegression:
    def __init__(self, regularization_strength: float = 1.0, data_regularization=True, 
                 # Keeping these for compatibility as requested, though unused in Closed-Form
                 learning_rate: float = None, epochs: int = None, early_stop_patience=None):
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
        I = jnp.eye(n_dims)
        
        # CRITICAL: Set the first diagonal element to 0. 
        # We do NOT want to penalize (shrink) the Bias/Intercept term.
        I = I.at[0, 0].set(0) 

        # 5. Compute X Transpose y
        XTy = jnp.dot(X_bias.T, y)

        # 6. Solve for Parameters (w)
        # (XTX + alpha * I) * w = XTy
        LHS = XTX + (self.regularization_strength * I)
        
        self.params = jnp.linalg.solve(LHS, XTy)
        
        # print("Fit complete using Closed-Form Solution.")

    def predict(self, X: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
        if self.params is None:
            raise ValueError("Model not fitted yet.")

        # 1. Scale Input
        if self.data_regularization:
            X = self.SS_X.transform(X)
            
        X = jnp.array(X)

        # 2. Add Intercept (Bias) Term
        X_bias = jnp.hstack([jnp.ones((X.shape[0], 1)), X])

        # 3. Compute Prediction
        # y = X * w
        y_pred = jnp.dot(X_bias, self.params)

        # 4. Return as Numpy Array
        return np.array(y_pred)
    

class LinearRegression:
    #self.params: jnp.ndarray

    def __init__(self, data_regularization = True):
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

    def predict(self, X) ->np.ndarray:
        if self.data_regularization:
            X = self.SS.transform(X)
        X = jnp.array(X)
        X = jnp.hstack([jnp.ones((X.shape[0], 1)), X])
        return np.array(jnp.dot(X, self.params))

    @staticmethod
    def evaluate(y_true, y_pred) -> float:
        return r2score(y_true, y_pred)




