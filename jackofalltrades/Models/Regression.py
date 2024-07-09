import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from typing import Union
import sklearn.metrics as metrics
import jax
import optax

import jackofalltrades.Errors.error


# Class for implementing Linear Regression
class LinearRegression:
    def __init__(self, learning_rate: float = 0.03, epochs: int = 10000, regularization_strength: float = 0.1,
                 data_regularization=True) -> None:
        """
        Initialize the LinearRegression object.

        Parameters:
        - learning_rate: Learning rate for gradient descent (default = 0.03).
        - epochs: Number of training iterations (default = 10000).
        - regularization_strength (default = 0.1)
        """
        self.n = None
        self.m = None
        try:
            self.cost = []
            self.epoch = []
            self.data_regularization = data_regularization
            self.regularization_strength = regularization_strength
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.w = None
            self.b = None
            self.params = {}
        except Exception as e:
            print("An error occurred during initialization:", str(e))

    def Standard(self, X):
        SS = StandardScaler()
        g = SS.fit_transform(np.array(X, dtype=np.float32))
        return jnp.array(g)

    def forward(self, X, params):
        return jnp.dot(X, params['w']) + params['b']

    def loss(self, params, X, y):
        y_pred = self.forward(X, params)
        return jnp.mean(jnp.square(y_pred - y))

    def update(self, params, grads):
        for key, value in grads.items():
            params[key] -= self.learning_rate * value
        return params

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], validation_split=0.2,
            early_stop_patience=5) -> None:
        """
        Train the linear regression model using gradient descent.
        Parameters:
        - X: Input features as a pandas DataFrame.
        - y: Target variable as a pandas Series.
        """
        try:
            X, y = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)
            if self.data_regularization:
                X = self.Standard(X)
            self.m, self.n = X.shape
            self.w = jnp.array(np.random.normal(size=self.n) * 1e-4)
            self.b = jnp.array(0.0)
            self.params = {'w': self.w, 'b': self.b}
            description = tqdm(range(self.epochs))
            X, X_test, y, y_test = train_test_split(X, y, test_size=validation_split, random_state=42)
            best_val_loss = float('inf')
            best_val_acc = float('-inf')
            patience = early_stop_patience
            solver = optax.adamw(learning_rate=0.003)
            opt_state = solver.init(self.params)
            if metrics.r2_score(y, self.forward(X, self.params)) < 0:
                tqdm.write("Negative R2Score, wait for a while")
            for i in description:

                acc = 0
                loss = 0
                description.set_description(f"R2Score:{metrics.r2_score(y, self.forward(X, self.params))}")
                for _ in range(10):
                    loss, grads = jax.value_and_grad(self.loss, argnums=0, allow_int=True)(self.params, X, y)
                    acc = round(metrics.r2_score(y, self.forward(X, self.params)), 5)
                    updates, opt_state = solver.update(grads, opt_state, self.params)
                    self.params = optax.apply_updates(self.params, updates)
                    # self.params = self.update(self.params, grads)
                    self.cost.append(loss)
                    self.epoch.append(i)

                if acc <= best_val_acc or loss >= best_val_loss:
                    patience -= 1
                else:
                    best_val_loss = loss
                    best_val_acc = acc
                    patience = early_stop_patience

                if patience == 0:
                    tqdm.write(f"Stopping early at epoch {i+1} due to constant or slow convergence rate")
                    if metrics.r2_score(y, self.forward(X, self.params)) < .5:
                        print('Try changing the hyperparameters')
                    description.close()
                    break
            if metrics.r2_score(y, self.forward(X, self.params)) <= .5:
                print("Model isn't working well try: ")
                print("1. Changing the Hyperparameters")
                print("2. Changing the Model e.x., MLPRegressor")

        except Exception as e:
            print("An error occurred during fitting:", str(e))

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict the target variable for the given input features.

        Parameters:
        - X_test: Input features for prediction as a pandas DataFrame.

        Returns:
        - Predicted target variable as a numpy array.
        """
        if self.data_regularization:
            X_test = self.Standard(X_test)
        return self.forward(X_test, self.params)

    def plot_cost(self) -> None:
        """
        Plot the cost function over training iterations.
        """
        plt.plot(self.cost, self.epoch)
        plt.show()

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Evaluate the model using the R-squared metric.

        Parameters:
        - X_test: Test input features as a numpy array.
        - y_test: Test target variable as a numpy array.
        """
        print(metrics.r2_score(y_true, y_pred))


class LogisticRegression:

    def __init__(self, learning_rate: float = 0.03, epochs: int = 10000, regularization_strength: float = 0.1,
                 data_regularization=True) -> None:
        """
        Initialize the LogisticRegression object.

        Parameters:
        - learning_rate: Learning rate for gradient descent (default = 0.03).
        - epochs: Number of training iterations (default = 10000).
        - regularization_strength (default = 0.1)
        """
        try:
            self.data_regularization = data_regularization
            self.regularization_strength = regularization_strength
            self.learning_rate = learning_rate
            self.epochs = epochs
        except Exception as e:
            print("An error occurred during initialization:", str(e))

    def Standard(self, X):
        SS = StandardScaler()
        g = SS.fit(X)
        g = SS.transform(X)
        return g

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate the sigmoid function.

        Parameters:
        - z: Input value.

        Returns:
        - Sigmoid value.
        """
        return 1 / (1 + jnp.exp(-z))

    def loss(self, w, b, X, y):
        y_pred = self.sigmoid(jnp.dot(X, w) + b)
        return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the logistic regression model using gradient descent.
        Parameters:
        - X: Input features as a pandas DataFrame.
        - y: Target variable as a pandas Series.
        """
        try:
            self.X, self.y = jnp.array(X, dtype=np.float32), jnp.array(y, dtype=np.float32).reshape(-1, 1)
            if self.data_regularization:
                self.X = self.Standard(self.X)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                    test_size=0.01, random_state=42)
            self.m, self.n = self.X_train.shape
            self.w = jnp.zeros((self.n, 1))
            self.b = jnp.zeros((1,))
            self.cost = []
            self.epoch = []
            description = tqdm(range(self.epochs))
            for i in description:
                loss_value = self.loss(self.w, self.b, self.X_train, self.y_train)
                grads = jax.grad(self.loss, argnums=(0, 1))(self.w, self.b, self.X_train, self.y_train)
                dw, db = grads
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
                self.cost.append(loss_value)
                self.epoch.append(i)
                description.set_description(f"Cost: {self.cost[-1]}")

        except Exception as e:
            print("An error occurred during fitting:", str(e))

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict the target variable for the given input features.

        Parameters:
        - X_test: Input features for prediction as a pandas DataFrame.

        Returns:
        - Predicted target variable as a numpy array.
        """
        if self.data_regularization:
            X_test = self.Standard(X_test)
        return (self.sigmoid(jnp.dot(np.array(X_test, dtype=jnp.float32) /
                                    jnp.max(self.X), self.w) + self.b) > 0.5).astype(int)

    def plot_cost(self) -> None:
        """
        Plot the cost function over training iterations.
        """
        plt.plot(self.epoch, self.cost)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Cost vs Epochs')
        plt.show()

    def evaluate(self, y_true: np.ndarray, y_predicted: np.ndarray) -> None:
        """
        Evaluate the model using the accuracy metric.

        Parameters:
        - y_true: True labels as a numpy array.
        - y_predicted: Predicted labels as a numpy array.
        """
        print(f"Accuracy: {metrics.accuracy_score(y_true, y_predicted):.2f}")


class MLPRegressor(tf.keras.Model):
    def __init__(self, learning_rate: float = 1.4e-4, epochs: int = 10000, regularization_strength: float = 0.1,
                 data_regularization=True, hidden_layers: int = 1) -> None:
        """
        Initialize the MLPRegressor object.

        Parameters:
        - learning_rate: Learning rate for gradient descent (default = 0.03).
        - epochs: Number of training iterations (default = 10000).
        - regularization_strength (default = 0.1)
        - data_regularization: Whether to standardize the input data (default = True).
        - hidden_layers: Number of hidden layers in the neural network (default = 1).
        - hidden_units: Number of units in each hidden layer (default = 10 per layer).
        - Architecture: [input_units, {hidden_units}, output_unit = 1]
        """

        try:
            self.y = None
            self.data_regularization = data_regularization
            self.regularization_strength = regularization_strength
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.hidden_layers = hidden_layers
            self.X = None
            self.cache_model: tf.keras.Model = None

        except Exception as e:
            print("An error occurred during initialization:", str(e))

    def Standard(self, X):
        SS = StandardScaler()
        SS.fit(self.X)
        g = SS.transform(X)
        return g

    def fit(self, X: Union[pd.DataFrame, np.ndarray, tf.Tensor], y: Union[pd.DataFrame, np.ndarray, tf.Tensor]) -> None:
        """
        Train the linear regression model using gradient descent.
        Parameters:
        - X: Input features as a pandas DataFrame or numpy array or tensorflow Tensor.
        - y: Target variable as a pandas Series or numpy array or tensorflow Tensor.
        """
        try:
            self.X = np.array(X, dtype=np.float32)
            self.y = np.array(y, dtype=np.float32)
            if self.data_regularization:
                self.X = self.Standard(self.X)
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.InputSpec(shape=(self.X.shape[1],)))
            for i in range(self.hidden_layers):
                model.add(tf.keras.layers.Dense(10, activation='relu',
                                                activity_regularizer=tf.keras.regularizers.l2(
                                                    l2=self.regularization_strength)))
                model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(1))
            model.compile(optimizer=tf.keras.optimizers.legacy.SGD(self.learning_rate),
                          loss=tf.keras.losses.MeanSquaredLogarithmicError(reduction='auto',
                                                                           name='mean_squared_logarithmic_error'))
            model.fit(self.X, self.y, epochs=self.epochs, verbose=1, batch_size=20)
            self.cache_model = model
        except Exception as e:
            print("An error occurred during fitting:", str(e))

    def predict(self, X_test: Union[pd.DataFrame, tf.Tensor, np.ndarray]) -> Union[np.ndarray, tf.Tensor]:
        """
        Predict the target variable for the given input features.

        Parameters:
        - X_test: Input features for prediction as a pandas DataFrame.

        Returns:
        - Predicted target variable as a numpy array.
        """
        X_test = np.array(X_test, dtype=np.float32)
        if self.data_regularization:
            X_test = self.Standard(X_test)
        return self.cache_model.predict(X_test)

    def evaluate(self, y_true: Union[pd.Series, tf.Tensor, np.ndarray],
                 y_pred: Union[pd.Series, tf.Tensor, np.ndarray]) -> None:
        """
        Evaluate the model using the R-squared metric.

        Parameters:
        - y_true: Test input features as a numpy array or pandas Series or tensorflow torch.
        - y_test: Test target variable as a numpy array.
        """
        self.cache_model.evaluate(y_true, y_pred)

    def save_model(self, path: str) -> None:
        """
        Save the trained model to a file.

        Parameters:
        - path: Path to save the model file.
        """
        super().save_model(path)
