"""from src.Models import LogisticRegression
import numpy as np
import pandas as pd
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

            lgtr = LogisticRegression(df[['Feature1', 'Feature2', 'Feature3', 'Feature4']], df['Output'])
            lgtr.fit()
            predictions = (lgtr.predict(testing_data[['Feature1', 'Feature2', 'Feature3', 'Feature4']]))
            self.assertEqual(list(predictions), list(testing_data['Output']))
            
      

if __name__ == '__main__':
      unittest.main()

            """

from test import LinearTest, LogisticTest
import unittest


if __name__ == '__main__':
    unittest.main()