import jax.numpy as np

class Error:
      """
      This class represents an error calculation for comparing true and predicted values.
      It provides various methods to calculate different types of errors.
      """

      def __init__(self, y_true, y_predicted) -> None:
            """
            Initializes an Error object with true and predicted values.

            Parameters:
            - y_true: The true values.
            - y_predicted: The predicted values.
            """
            self.y_true, self.y_predicted = np.array(y_true, dtype=np.float32), np.array(y_predicted, dtype=np.float32)

      def MSE(self) -> np.float32:
            """
            Calculates the Mean Squared Error (MSE) between true and predicted values.

            Returns:
            - The MSE value as a numpy float32.
            """
            return np.mean(np.square((self.y_true - self.y_predicted)))

      def RMSE(self) -> np.float32:
            """
            Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

            Returns:
            - The RMSE value as a numpy float32.
            """
            return np.sqrt(np.mean(np.square((self.y_true - self.y_predicted))))

      def MAE(self) -> np.float32:
            """
            Calculates the Mean Absolute Error (MAE) between true and predicted values.

            Returns:
            - The MAE value as a numpy float32.
            """
            return np.mean(np.abs((self.y_true - self.y_predicted)))

      def SOAE(self) -> np.float32:
            """
            Calculates the Sum of Absolute Errors (SOAE) between true and predicted values.

            Returns:
            - The SOAE value as a numpy float32.
            """
            return np.abs(np.sum(self.y_true - self.y_predicted))

      def SOE(self) -> np.float32:
            """
            Calculates the Sum of Errors (SOE) between true and predicted values.

            Returns:
            - The SOE value as a numpy float32.
            """
            return np.sum(self.y_true - self.y_predicted)