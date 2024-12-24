import pandas as pd
import sklearn.datasets as datasets
import numpy as np
import importlib.resources as pkg_resources


def get_real_estate():
    with pkg_resources.open_text('jackofalltrades.datasets', 'Real estate.csv') as file:
        df = pd.read_csv(file)
    return df.drop(['Y house price of unit area'], axis=1), df['Y house price of unit area']


def get_california_housing():
    housing = datasets.fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    return X, y


def get_bitcoin():
    with pkg_resources.open_text('jackofalltrades.datasets', 'BTC-USD.csv') as file:
        df = pd.read_csv(file)
    cols = df.columns
    for i in cols:
        if df[i].dtype==object:
            df.drop(columns=[i], inplace=True)
        else:
            df[i].fillna(np.mean(df[i]))
    return df.drop(columns=['Adj Close']), df['Adj Close']


def get_london_housing():
    with pkg_resources.open_text('jackofalltrades.datasets', 'london_house_prices.csv') as file:
        df = pd.read_csv(file)
    cols = df.columns
    for i in cols:
        if df[i].dtype == object:
            df.drop(columns=[i], inplace=True)
        else:
            df[i].fillna(np.mean(df[i]))
    return df.drop(columns=['price_pounds']), df['price_pounds']


def get_fuels_data():
    with pkg_resources.open_text('jackofalltrades.datasets', 'all_fuels_data.csv') as file:
        df = pd.read_csv(file)
    cols = df.columns
    for i in cols:
        if df[i].dtype==object:
            df.drop(columns=[i], inplace=True)
        else:
            df[i].fillna(np.mean(df[i]))
    return df.drop(columns=['close']), df['close']


def get_mnist():
    mnist = datasets.fetch_openml('mnist_784')
    X = pd.DataFrame(mnist.data)
    y = pd.Series(mnist.target, dtype=int)
    return X, y


def get_boston_housing():
    boston = datasets.load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target)
    return X, y


def get_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y


def get_breast_cancer():
    bc = datasets.load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target, dtype=int)
    return X, y
