import pandas as pd
import sklearn.datasets as datasets 
import numpy as np


class get_dataset:

      def get_real_estate(self):
            df = pd.read_csv(r'jackofalltradespy\\datasets\\Real estate.csv')
            return df.drop(['Y house price of unit area'], axis=1), df['Y house price of unit area']
            
      def get_california_housing(self):
            self.housing = datasets.fetch_california_housing()
            self.X = pd.DataFrame(self.housing.data, columns=self.housing.feature_names)
            self.y = pd.Series(self.housing.target)
            return self.X, self.y
      
      def get_bitcoin(self):
            df = pd.read_csv(r'jackofalltradespy\\datasets\\BTC-USD.csv')
            cols = df.columns
            print(df.shape)
            for i in cols:
                  if df[i].dtype == object:
                        df.drop(columns=[i], inplace=True)
                  else:
                        df[i].fillna(np.mean(df[i]))
            return df.drop(columns=['Adj Close']), df['Adj Close']
      
      def get_london_housing(self):
            df = pd.read_csv(r'jackofalltradespy\\datasets\\london_house_prices.csv')
            cols = df.columns
            for i in cols:
                  if df[i].dtype == object:
                        df.drop(columns=[i], inplace=True)
                  else:
                        df[i].fillna(np.mean(df[i]))
            return df.drop(columns=['price_pounds']), df['price_pounds']
      
      def get_fuels_data(self):
            df = pd.read_csv('jackofalltradespy\\datasets\\all_fuels_data.csv')
            cols = df.columns
            for i in cols:
                  if df[i].dtype == object:
                        df.drop(columns=[i], inplace=True)
                  else:
                        df[i].fillna(np.mean(df[i]))
            return df.drop(columns=['close']), df['close']

      

def main():
      gd = get_dataset()
      X= gd.get_fuels_data()
      print(X.columns)
      cols = X.columns
      for i in cols:
            print(X[i].dtype)
      
if __name__ == "__main__":
      main()