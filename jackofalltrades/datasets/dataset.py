import pandas as pd
import sklearn.datasets as datasets
import numpy as np
import importlib.resources as pkg_resources


class get_dataset:

    def __init__(self):
        self.X = None
        self.y = None
        self.df = None

    def get_real_estate(self):
        with pkg_resources.open_text('jackofalltrades.datasets', 'Real estate.csv') as file:
            self.df = pd.read_csv(file)
        return self.df.drop(['Y house price of unit area'], axis=1), self.df['Y house price of unit area']

    def get_california_housing(self):
        housing = datasets.fetch_california_housing()
        self.X = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.y = pd.Series(housing.target)
        return self.X, self.y

    def get_bitcoin(self):
        with pkg_resources.open_text('jackofalltrades.datasets', 'BTC-USD.csv') as file:
            self.df = pd.read_csv(file)
        cols = self.df.columns
        for i in cols:
            if self.df[i].dtype==object:
                self.df.drop(columns=[i], inplace=True)
            else:
                self.df[i].fillna(np.mean(self.df[i]))
        return self.df.drop(columns=['Adj Close']), self.df['Adj Close']

    def get_london_housing(self):
        with pkg_resources.open_text('jackofalltrades.datasets', 'london_house_prices.csv') as file:
            self.df = pd.read_csv(file)
        cols = self.df.columns
        for i in cols:
            if self.df[i].dtype == object:
                self.df.drop(columns=[i], inplace=True)
            else:
                self.df[i].fillna(np.mean(self.df[i]))
        return self.df.drop(columns=['price_pounds']), self.df['price_pounds']

    def get_fuels_data(self):
        with pkg_resources.open_text('jackofalltrades.datasets', 'all_fuels_data.csv') as file:
            self.df = pd.read_csv(file)
        cols = self.df.columns
        for i in cols:
            if self.df[i].dtype==object:
                self.df.drop(columns=[i], inplace=True)
            else:
                self.df[i].fillna(np.mean(self.df[i]))
        return self.df.drop(columns=['close']), self.df['close']

    def get_mnist(self):
        mnist = datasets.fetch_openml('mnist_784')
        self.X = pd.DataFrame(mnist.data)
        self.y = pd.Series(mnist.target, dtype=int)
        return self.X, self.y

    def get_boston_housing(self):
        boston = datasets.load_boston()
        self.X = pd.DataFrame(boston.data, columns=boston.feature_names)
        self.y = pd.Series(boston.target)
        return self.X, self.y

    def get_iris(self):
        iris = datasets.load_iris()
        self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.y = pd.Series(iris.target)
        return self.X, self.y

    def get_breast_cancer(self):
        bc = datasets.load_breast_cancer()
        self.X = pd.DataFrame(bc.data, columns=bc.feature_names)
        self.y = pd.Series(bc.target, dtype=int)
        return self.X, self.y
