
import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models import LogisticRegression , LinearRegression
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SLogisticRegression
from sklearn.linear_model import LinearRegression as SLinearRegression
import unittest
class LogisticTest(unittest.TestCase):
      def test(self):
      # Create a pandas DataFrame dataset
            df = pd.DataFrame({
                  'Feature1': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  'Feature2': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
                  'Feature3': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                  'Feature4': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                  'Output': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            })

            # Create testing data
            testing_data = pd.DataFrame({
                  'Feature1': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  'Feature2': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                  'Feature3': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                  'Feature4': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                  'Output': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            })

            lgtr = SLogisticRegression()
            lgtr.fit(df[['Feature1', 'Feature2', 'Feature3', 'Feature4']], df['Output'])
            predictions = (lgtr.predict(testing_data[['Feature1', 'Feature2', 'Feature3', 'Feature4']]))
            print("Testing Logistic Regression")
            print(list(predictions))
            print(list(testing_data['Output']))
            self.assertEqual(list(predictions), list(testing_data['Output']))
            
      
class LinearTest(unittest.TestCase):
      
      def test(self):
        # Create a pandas DataFrame dataset
            df = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'Feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
            'Feature3': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60],
            'Feature4': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80],
            'Output': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            })

        # Create testing data
            testing_data = pd.DataFrame({
                  'Feature1': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                  'Feature2': [42, 44, 46, 48, 50, 52, 54, 56, 58, 60],
                  'Feature3': [63, 66, 69, 72, 75, 78, 81, 84, 87, 90],
                  'Feature4': [84, 88, 92, 96, 100, 104, 108, 112, 116, 120],
                  'Output': [105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
            })

            lnr = SLinearRegression()
            lnr.fit(df[['Feature1', 'Feature2', 'Feature3', 'Feature4']], df['Output'])

            predictions = (lnr.predict(testing_data[['Feature1', 'Feature2', 'Feature3', 'Feature4']]))
            vfunc = np.vectorize(lambda x: int(x) + 1 if x % 1 > 0.5 else int(x))
            #rfunct = np.vectorize(lambda x: round(float(x), 3))
            print("Testing Linear Regression")
            print(list((predictions)))
            print(list(testing_data['Output']))
            #print(lnr.evaluate(testing_data['Output'], predictions))
            self.assertEqual(list(vfunc(predictions)), list(testing_data['Output']))

if __name__ == '__main__':
      unittest.main()