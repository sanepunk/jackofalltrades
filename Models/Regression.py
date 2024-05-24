import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler

import sklearn.metrics as metrics

# Class for implementing Linear Regression
class LinearRegression:
      def __init__(self, X: pd.DataFrame , y: pd.Series, learning_rate : float = 0.03, epochs : int = 10000, regularization_strength: float = 0.1) -> None:
            """
            Initialize the LinearRegression object.

            Parameters:
            - X: Input features as a pandas DataFrame.
            - y: Target variable as a pandas Series.
            - learning_rate: Learning rate for gradient descent (default = 0.03).
            - epochs: Number of training iterations (default = 10000).
            """
            try:
                  self.regularization_strength = regularization_strength
                  self.X, self.y = np.array(X, dtype= np.float32), np.array(y, dtype = np.float32)
                  self.X = self.Standard(self.X)
                  self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                  self.learning_rate = learning_rate
                  self.epochs = epochs
            except Exception as e:
                  print("An error occurred during initialization:", str(e))
      
      def Standard(self, X):
            SS = StandardScaler()
            g = SS.fit(X)
            g = SS.transform(X)
            return g
      def fit(self) -> None:
            """
            Train the linear regression model using gradient descent.
            """
            try:
                  self.m, self.n = self.X_train.shape
                  self.w = np.random.normal(size=self.n)
                  self.b = 0
                  y_pred = np.dot(self.X_train, self.w) + self.b
                  self.cost = []
                  self.epoch = []
                  description = tqdm(range(self.epochs))
                  for i in description:
                        y_pred = np.dot(self.X_train, self.w) + self.b
                        dw = (1/self.m) * np.dot(self.X_train.T, (y_pred - self.y_train)) + (2 * self.regularization_strength / self.m) * self.w
                        db = (1/self.m) * np.sum(y_pred - self.y_train)
                        self.w -= self.learning_rate * dw
                        self.b -= self.learning_rate * db
                        self.cost.append(np.mean(np.square(y_pred - self.y_train)))
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

            X_test = self.Standard(X_test)
            return np.dot(np.array(X_test, dtype = np.float32), self.w) + self.b

      def plot_cost(self) -> None:
            """
            Plot the cost function over training iterations.
            """
            plt.plot(self.cost ,self.epoch)
            plt.show()

      def evaluate(self, y_true : np.ndarray, y_pred : np.ndarray) -> None:
            """
            Evaluate the model using the R-squared metric.

            Parameters:
            - X_test: Test input features as a numpy array.
            - y_test: Test target variable as a numpy array.
            """
            print(metrics.r2_score(y_true, y_pred))

class LogisticRegression:

      def __init__(self, X: pd.DataFrame , y: pd.Series, learning_rate : float = 0.03, epochs : int = 10000, regularization_strength: float = 0.1) -> None:
            """
            Initialize the LogisticRegression object.

            Parameters:
            - X: Input features as a pandas DataFrame.
            - y: Target variable as a pandas Series.
            - learning_rate: Learning rate for gradient descent (default = 0.03).
            - epochs: Number of training iterations (default = 10000).
            """
            try:
                  self.regularization_strength = regularization_strength
                  self.X, self.y = np.array(X, dtype= np.float32), np.array(y, dtype = np.float32)
                  self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X / np.max(self.X), self.y, test_size=0.2, random_state=42)
                  self.learning_rate = learning_rate
                  self.epochs = epochs
            except Exception as e:
                  print("An error occurred during initialization:", str(e))   

      def sigmoid(self, z: np.ndarray) -> np.ndarray:
            """
            Calculate the sigmoid function.

            Parameters:
            - z: Input value.

            Returns:
            - Sigmoid value.
            """
            return 1 / (1 + np.exp(-z))
      
      def fit(self) -> None:
            """   
            Train the logistic regression model using gradient descent.
            """
            try:
                  self.m, self.n = self.X_train.shape
                  self.w = np.zeros(self.n)
                  self.b = 0
                  y_pred = self.sigmoid(np.dot(self.X_train, self.w) + self.b)
                  self.cost = []
                  self.epoch = []
                  description = tqdm(range(self.epochs))
                  for i in description:
                        y_pred = self.sigmoid(np.dot(self.X_train, self.w) + self.b)
                        loss = -1/self.m * np.sum(self.y_train * np.log(y_pred) + (1 - self.y_train) * np.log(1 - y_pred))
                        dw = (1/self.m) * np.dot(self.X_train.T, (y_pred - self.y_train)) + (2 * self.regularization_strength / self.m) * self.w
                        db = (1/self.m) * np.sum(y_pred - self.y_train)
                        self.w -= self.learning_rate * dw
                        self.b -= self.learning_rate * db
                        self.cost.append(loss)
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
            return (self.sigmoid(np.dot(np.array(X_test , dtype = np.float32) / np.max(self.X), self.w) + self.b) > 0.5).astype(int)

      def plot_cost(self) -> None:
            """
            Plot the cost function over training iterations.
            """
            plt.plot(self.cost ,self.epoch)
            plt.show()

      def evaluate(self, y_true : np.ndarray, y_predicted : np.ndarray) -> None:
            """
            Evaluate the model using the R-squared metric.

            Parameters:
            - X_test: Test input features as a numpy array.
            - y_test: Test target variable as a numpy array.
            """
            print(metrics.accuracy_score(y_true, (y_predicted > 0.5).astype(int)))