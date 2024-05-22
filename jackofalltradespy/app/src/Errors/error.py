import jax.numpy as np
class Error:
      """
      This 
      """
      def __init__(self, y_true, y_predicted) -> None:
            self.y_true, self.y_predicted = np.array(y_true, dtype=np.float32), np.array(y_predicted, dtype=np.float32)
            
      def MSE(self) -> np.float32:
            return np.mean(np.square((self.y_true - self.y_predicted)))
      
      def RMSE(self) -> np.float32:
            return np.sqrt(np.mean(np.square((self.y_true - self.y_predicted))))
      
      def MAE(self) -> np.float32:
            return np.mean(np.abs((self.y_true - self.y_predicted)))
      
      def SOAE(self) -> np.float32:
            return np.abs(np.sum(self.y_true - self.y_predicted))
      
      def SOE(self) -> np.float32:
            return np.sum(self.y_true - self.y_predicted)
      
      

            