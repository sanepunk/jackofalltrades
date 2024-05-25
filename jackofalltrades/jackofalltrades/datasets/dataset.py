import pandas as pd
import sklearn.datasets as datasets 
import numpy as np
import pkg_resources

class get_dataset:

      def get_real_estate(self):
            file_path = pkg_resources.resource_filename('jackofalltrades', 'datasets/Real estate.csv')
            df = pd.read_csv(file_path)
            return df.drop(['Y house price of unit area'], axis=1), df['Y house price of unit area']
            
      def get_california_housing(self):
            self.housing = datasets.fetch_california_housing()
            self.X = pd.DataFrame(self.housing.data, columns=self.housing.feature_names)
            self.y = pd.Series(self.housing.target)
            return self.X, self.y
      
      def get_bitcoin(self):
            file_path = pkg_resources.resource_filename('jackofalltrades', 'datasets/BTC-USD.csv')
            df = pd.read_csv(file_path)
            cols = df.columns
            for i in cols:
                  if df[i].dtype == object:
                        df.drop(columns=[i], inplace=True)
                  else:
                        df[i].fillna(np.mean(df[i]))
            return df.drop(columns=['Adj Close']), df['Adj Close']
      
      def get_london_housing(self):
            file_path = pkg_resources.resource_filename('jackofalltrades', 'datasets/london_house_prices.csv')
            df = pd.read_csv(file_path)
            cols = df.columns
            for i in cols:
                  if df[i].dtype == object:
                        df.drop(columns=[i], inplace=True)
                  else:
                        df[i].fillna(np.mean(df[i]))
            return df.drop(columns=['price_pounds']), df['price_pounds']
      
      def get_fuels_data(self):
            file_path = pkg_resources.resource_filename('jackofalltrades', 'datasets/all_fuels_data.csv')
            df = pd.read_csv(file_path)
            cols = df.columns
            for i in cols:
                  if df[i].dtype == object:
                        df.drop(columns=[i], inplace=True)
                  else:
                        df[i].fillna(np.mean(df[i]))
            return df.drop(columns=['close']), df['close']
