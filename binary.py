from jackofalltradespy.Models import LogisticRegression
from sklearn.model_selection import train_test_split
from jackofalltradespy.Errors import Error
from sklearn.linear_model import LogisticRegression as SLGR
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
      # Load Iris dataset (multi-class classification)
      iris = load_iris()
      X_iris, y_iris = iris.data, iris.target

      # Load Breast Cancer dataset (binary classification)
      bc = load_breast_cancer()
      X_bc, y_bc = bc.data, bc.target
      # Split data into training and testing sets for Iris dataset
      X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

      # Split data for Breast Cancer dataset
      X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

      gg = LogisticRegression(X_train_bc, y_train_bc)
      gg.fit()
      gg.evaluate(y_test_bc, gg.predict(X_test_bc))
      er = Error(y_test_bc, gg.predict(X_test_bc))
      print(er.Accuracy())
      print(er.Cross_Entropy())
      print(er.F1Score())

main()