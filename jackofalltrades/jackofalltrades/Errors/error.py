import jax.numpy as np
from sklearn.metrics import confusion_matrix

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
            
      def MSE(self) -> np.float32:
            """
            Calculate the Mean Squared Error (MSE).

            Returns:
                  np.float32: The calculated MSE value.
            """
            try:
                  return np.mean(np.square((self.y_true - self.y_predicted)))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def RMSE(self) -> np.float32:
            """
            Calculate the Root Mean Squared Error (RMSE).

            Returns:
                  np.float32: The calculated RMSE value.
            """
            try:
                  return np.sqrt(np.mean(np.square((self.y_true - self.y_predicted))))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def MAE(self) -> np.float32:
            """
            Calculate the Mean Absolute Error (MAE).

            Returns:
                  np.float32: The calculated MAE value.
            """
            try:
                  return np.mean(np.abs((self.y_true - self.y_predicted)))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def SOAE(self) -> np.float32:
            """
            Calculate the Sum of Absolute Errors (SOAE).

            Returns:
                  np.float32: The calculated SOAE value.
            """
            try:
                  return np.abs(np.sum(self.y_true - self.y_predicted))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def SOE(self) -> np.float32:
            """
            Calculate the Sum of Errors (SOE).

            Returns:
                  np.float32: The calculated SOE value.
            """
            try:
                  return np.sum(self.y_true - self.y_predicted)
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def RSquared(self) -> np.float32:
            """
            Calculate the R-squared value.

            Returns:
                  np.float32: The calculated R-squared value.
            """
            try:
                  return 1 - (np.sum(np.square(self.y_true - self.y_predicted)) / np.sum(np.square(self.y_true - np.mean(self.y_true))))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def MAPE(self) -> np.float32:
            """
            Calculate the Mean Absolute Percentage Error (MAPE).

            Returns:
                  np.float32: The calculated MAPE value.
            """
            try:
                  return np.mean(np.abs((self.y_true - self.y_predicted) / self.y_true)) * 100
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def AdjustedRSquared(self, n, p) -> np.float32:
            """
            Calculate the Adjusted R-squared value.

            Args:
                  n (int): Number of observations.
                  p (int): Number of predictors.

            Returns:
                  np.float32: The calculated Adjusted R-squared value.
            """
            try:
                  return 1 - ((1 - self.RSquared()) * (n - 1) / (n - p - 1))
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def Accuracy(self) -> np.float32:
            """
            Calculate the Accuracy.

            Returns:
                  np.float32: The calculated Accuracy value.
            """
            try:
                  cm = confusion_matrix(self.y_true, self.y_predicted)
                  TN, FP, FN, TP = cm.ravel()
                  return (TP + TN) / (TP + TN + FP + FN)
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def Precision(self) -> np.float32:
            """
            Calculate the Precision.

            Returns:
                  np.float32: The calculated Precision value.
            """
            try:
                  cm = confusion_matrix(self.y_true, self.y_predicted)
                  TN, FP, FN, TP = cm.ravel()
                  return TP / (TP + FP)
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def Recall(self) -> np.float32:
            """
            Calculate the Recall.

            Returns:
                  np.float32: The calculated Recall value.
            """
            try:
                  cm = confusion_matrix(self.y_true, self.y_predicted)
                  TN, FP, FN, TP = cm.ravel()
                  return TP / (TP + FN) 
            except Exception as e:
                  raise Exception(f"Error: {e}")  
  
      def F1Score(self) -> np.float32:
            """
            Calculate the F1 Score.

            Returns:
                  np.float32: The calculated F1 Score value.
            """
            try:
                  cm = confusion_matrix(self.y_true, self.y_predicted)
                  TN, FP, FN, TP = cm.ravel()
                  return 2 * TP / (2 * TP + FP + FN)
            except Exception as e:
                  raise Exception(f"Error: {e}")
  
      def Cross_Entropy(self) -> np.float32:
            """
            Calculate the Cross Entropy.

            Returns:
                  np.float32: The calculated Cross Entropy value.
            """
            try:
                  return (1 - self.y_true) * np.log(1 - self.y_predicted) + self.y_true * np.log(self.y_predicted)
            except Exception as e:  # Use the actual exception object 'e'
                  # Consider a more informative error message or handling strategy
                  print(f"Error during Cross Entropy calculation: {e}")
                  return np.nan  # Or a more appropriate default value

            
      

            