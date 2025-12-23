import jax.numpy as np
from sklearn.metrics import confusion_matrix


def r2score(y_true ,y_pred):
      """
      Calculate the R-squared (Coefficient of Determination) between true and predicted values.

      R-squared measures the proportion of the variance in the dependent variable
      that is predictable from the independent variables. It is calculated as:

            R² = 1 - (SS_res / SS_tot)

      where:
      - SS_res (Residual Sum of Squares) = Σ(y_true - y_pred)²
      - SS_tot (Total Sum of Squares) = Σ(y_true - mean(y_true))²

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.

      Returns:
      - R-squared value.
      """
      return 1 - (np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))

def accuracy(y_true, y_pred):
      """
      Calculate the accuracy between true and predicted values.

      Accuracy is the ratio of correctly predicted observations to the total observations.
      It is calculated as:

            Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.

      Returns:
      - Accuracy value.
      """
      return np.mean(y_pred == y_true)

def mse(y_true, y_pred):
      """
      Calculate the Mean Squared Error (MSE) between true and predicted values.

      MSE measures the average of the squares of the errors—that is, the average squared
      difference between the estimated values and the actual value. It is calculated as:

            MSE = (1/n) * Σ(y_true - y_pred)²

      where n is the number of observations.

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.

      Returns:
      - Mean Squared Error value.
      """
      return np.mean(np.square((y_true - y_pred)))

def rmse(y_true, y_pred):
      """
      Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

      RMSE is the square root of the average of squared differences between prediction and
      actual observation. It is calculated as:

            RMSE = sqrt(MSE)
                  = sqrt((1/n) * Σ(y_true - y_pred)²)

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.

      Returns:
      - Root Mean Squared Error value.
      """
      return np.sqrt(np.mean(np.square((y_true - y_pred))))

def mae(y_true, y_pred):
      """
      Calculate the Mean Absolute Error (MAE) between true and predicted values.

      MAE measures the average magnitude of the errors in a set of predictions, without
      considering their direction. It is calculated as:

            MAE = (1/n) * Σ|y_true - y_pred|

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.

      Returns:
      - Mean Absolute Error value.
      """
      return np.mean(np.abs((y_true - y_pred)))

def soae(y_true, y_pred):
      """
      Calculate the Sum of Absolute Errors (SOAE) between true and predicted values.

      SOAE measures the total absolute difference between the predicted and actual values.
      It is calculated as:

            SOAE = Σ|y_true - y_pred|

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.

      Returns:
      - Sum of Absolute Errors value.
      """
      return np.abs(np.sum(y_true - y_pred))

def soe(y_true, y_pred):
      """
      Calculate the Sum of Errors (SOE) between true and predicted values.

      SOE measures the total difference between the predicted and actual values.
      It is calculated as:

            SOE = Σ(y_true - y_pred)

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.

      Returns:
      - Sum of Errors value.
      """
      return np.sum(y_true - y_pred)

def mape(y_true, y_pred):
      """
      Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

      MAPE measures the average magnitude of errors in a set of predictions, expressed as
      a percentage of the actual values. It is calculated as:

            MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.

      Returns:
      - Mean Absolute Percentage Error value.
      """
      return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adjusted_r2score(y_true, y_pred, n, p):
      """
      Calculate the Adjusted R-squared between true and predicted values.

      Adjusted R-squared adjusts the R-squared value based on the number of predictors
      in the model, providing a more accurate measure for multiple regression. It is
      calculated as:

            Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]

      where:
      - R² is the R-squared value.
      - n is the number of observations.
      - p is the number of predictors.

      Parameters:
      - y_true: Actual values.
      - y_pred: Predicted values.
      - n: Number of observations.
      - p: Number of predictors.

      Returns:
      - Adjusted R-squared value.
      """
      return 1 - ((1 - r2score(y_true, y_pred)) * (n - 1) / (n - p - 1))

def precision(y_true, y_pred):
      """
      Calculate the precision between true and predicted binary classifications.

      Precision is the ratio of correctly predicted positive observations to the total
      predicted positives. It is calculated as:

            Precision = TP / (TP + FP)

      where:
      - TP is the number of true positives.
      - FP is the number of false positives.

      Parameters:
      - y_true: Actual binary values.
      - y_pred: Predicted binary values.

      Returns:
      - Precision value.
      """
      cm = confusion_matrix(y_true, y_pred)
      TN, FP, FN, TP = cm.ravel()
      return TP / (TP + FP)

def recall(y_true, y_pred):
      """
      Calculate the recall (sensitivity) between true and predicted binary classifications.

      Recall is the ratio of correctly predicted positive observations to all observations
      in the actual class. It is calculated as:

            Recall = TP / (TP + FN)

      where:
      - TP is the number of true positives.
      - FN is the number of false negatives.

      Parameters:
      - y_true: Actual binary values.
      - y_pred: Predicted binary values.

      Returns:
      - Recall value.
      """
      cm = confusion_matrix(y_true, y_pred)
      TN, FP, FN, TP = cm.ravel()
      return TP / (TP + FN)

def f1score(y_true, y_pred):
      """
      Calculate the F1 Score between true and predicted values.

      The F1 Score is the harmonic mean of precision and recall, providing a balance
      between the two, especially useful for imbalanced datasets. It is calculated as:

            F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

      where:
      - Precision = TP / (TP + FP)
      - Recall = TP / (TP + FN)
      - TP: True Positives
      - FP: False Positives
      - FN: False Negatives

      Parameters:
      - y_true: Actual binary labels.
      - y_pred: Predicted binary labels.

      Returns:
      - F1 Score value.
      """
      cm = confusion_matrix(y_true, y_pred)
      TN, FP, FN, TP = cm.ravel()
      return 2 * TP / (2 * TP + FP + FN)

def cross_entropy(y_true, y_pred):
      """
      Calculate the Cross-Entropy Loss between true and predicted probabilities.

      Cross-Entropy Loss, also known as Log Loss, measures the performance of a classification
      model whose output is a probability value between 0 and 1. It quantifies the difference
      between two probability distributions: the true labels and the predicted probabilities.
      It is calculated as:

            Cross-Entropy Loss = -Σ [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]

      where:
      - y_true: Actual binary labels (0 or 1).
      - y_pred: Predicted probabilities for the positive class (values between 0 and 1).

      Parameters:
      - y_true: Actual binary labels.
      - y_pred: Predicted probabilities.

      Returns:
      - Cross-Entropy Loss value.
      """
      return (1 - y_true) * np.log(1 - y_pred) + y_true * np.log(y_pred)


class Error:
      """
      Class to calculate various error metrics for regression and classification tasks.
      """

      def __init__(self, y_true, y_predicted) -> None:
            """
            Initialize the Error class with true and predicted values.

            Args:
                  y_true (array-like): True values.
                  y_predicted (array-like): Predicted values.
            """
            try:
                  self.y_true, self.y_predicted = np.array(y_true, dtype=np.float32), np.array(y_predicted, dtype=np.float32)
            except Exception as e:
                  raise Exception(f"Error: {e}")
            
      def MSE(self) -> np.array:
            """
            Calculate the Mean Squared Error (MSE).

            Returns:
                  np.array: The calculated MSE value.
            """
            try:
                  return np.mean(np.square((self.y_true - self.y_predicted)))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def RMSE(self) -> np.array:
            """
            Calculate the Root Mean Squared Error (RMSE).

            Returns:
                  np.array: The calculated RMSE value.
            """
            try:
                  return np.sqrt(np.mean(np.square((self.y_true - self.y_predicted))))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def MAE(self) -> np.array:
            """
            Calculate the Mean Absolute Error (MAE).

            Returns:
                  np.array: The calculated MAE value.
            """
            try:
                  return np.mean(np.abs((self.y_true - self.y_predicted)))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def SOAE(self) -> np.array:
            """
            Calculate the Sum of Absolute Errors (SOAE).

            Returns:
                  np.array: The calculated SOAE value.
            """
            try:
                  return np.abs(np.sum(self.y_true - self.y_predicted))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def SOE(self) -> np.array:
            """
            Calculate the Sum of Errors (SOE).

            Returns:
                  np.array: The calculated SOE value.
            """
            try:
                  return np.sum(self.y_true - self.y_predicted)
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def RSquared(self) -> np.array:
            """
            Calculate the R-squared value.

            Returns:
                  np.array: The calculated R-squared value.
            """
            try:
                  return 1 - (np.sum(np.square(self.y_true - self.y_predicted)) / np.sum(np.square(self.y_true - np.mean(self.y_true))))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def MAPE(self) -> np.array:
            """
            Calculate the Mean Absolute Percentage Error (MAPE).

            Returns:
                  np.array: The calculated MAPE value.
            """
            try:
                  return np.mean(np.abs((self.y_true - self.y_predicted) / self.y_true)) * 100
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def AdjustedRSquared(self, n, p) -> np.array:
            """
            Calculate the Adjusted R-squared value.

            Args:
                  n (int): Number of observations.
                  p (int): Number of predictors.

            Returns:
                  np.array: The calculated Adjusted R-squared value.
            """
            try:
                  return 1 - ((1 - self.RSquared()) * (n - 1) / (n - p - 1))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def Accuracy(self) -> np.array:
            """
            Calculate the Accuracy.

            Returns:
                  np.array: The calculated Accuracy value.
            """
            try:
                  cm = confusion_matrix(self.y_true, self.y_predicted)
                  TN, FP, FN, TP = cm.ravel()
                  return (TP + TN) / (TP + TN + FP + FN)
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def Precision(self) -> np.array:
            """
            Calculate the Precision.

            Returns:
                  np.array: The calculated Precision value.
            """
            try:
                  cm = confusion_matrix(self.y_true, self.y_predicted)
                  TN, FP, FN, TP = cm.ravel()
                  return TP / (TP + FP)
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def Recall(self) -> np.array:
            """
            Calculate the Recall.

            Returns:
                  np.array: The calculated Recall value.
            """
            try:
                  cm = confusion_matrix(self.y_true, self.y_predicted)
                  TN, FP, FN, TP = cm.ravel()
                  return TP / (TP + FN) 
            except Exception as e:
                  raise Exception(f"Error: {e}")  
  
      def F1Score(self) -> np.array:
            """
            Calculate the F1 Score.

            Returns:
                  np.array: The calculated F1 Score value.
            """
            try:
                  cm = confusion_matrix(self.y_true, self.y_predicted)
                  TN, FP, FN, TP = cm.ravel()
                  return 2 * TP / (2 * TP + FP + FN)
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def Cross_Entropy(self) -> np.array:
            """
            Calculate the Cross Entropy.

            Returns:
                  np.array: The calculated Cross Entropy value.
            """
            try:
                  return (1 - self.y_true) * np.log(1 - self.y_predicted) + self.y_true * np.log(self.y_predicted)
            except Exception as e:  # Use the actual exception object 'e'
                  # Consider a more informative error message or handling strategy
                  print(f"Error during Cross Entropy calculation: {e}")
                  return np.nan  # Or a more appropriate default value

            
      

            