import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_X(df): 
    """Add bias unit to dataset and return as a whole"""
    ones = pd.DataFrame({'ones': np.ones(len(df))}) # add bias unit
    data = pd.concat([ones, df], axis=1)  # concatenate bias feature and other features
    return data.iloc[:, :-1].values  # this line of code returns ndarray, not matrix


def get_y(df):
    """Get predictions of the dataset, assume that the last column is the target"""  
    return np.array(df.iloc[:, -1]) # df.iloc[:, -1] means the last column of df


def normalize_feature(df):
    """Apply normalization to all columns"""
    return df.apply(lambda column: (column - column.mean()) / column.std()) # apply standard score 


def cost(theta, X, y):
    """
    Calculate the cost of given parameters
    
    Args:
        theta : R(n), parameters for linear regression
        X: R(m*n), m number of samples, n number of features
        y: R(m) predictions
    """
    m = X.shape[0] # number of samples

    inner = X @ theta - y  # R(m*1)，X @ theta is equivalent to X.dot(theta)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost


def gradient(theta, X, y):
    """
    Calculate the gradient of each parameter
    """
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta is equivalent to X.dot(theta)

    return inner / m

def batch_gradient_descent(theta, X, y, epoch, alpha=0.01):
    """
    Fit linear regression using batch gradient descent algorithm,
    return gradient and cost

    Args:
        theta : R(n), parameters for linear regression
        X: R(m*n), m number of samples, n number of features
        y: R(m) predictions
        epoch: number of iterations
        alpha: learning rate. default 0.01
    """
    cost_data = [cost(theta, X, y)]
    _theta = theta.copy()  # make a copy to avoid confusion

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(cost(_theta, X, y))

    return _theta, cost_data


def stochastic_gradient_descent(theta, X, y, epoch, alpha=0.01):
    """
    Fit linear regression using statistic gradient descent algorithm,
    return gradient and cost

    Args:
        theta : R(n), parameters for linear regression
        X: R(m*n), m number of samples, n number of features
        y: R(m) predictions
        epoch: number of iterations
        alpha: learning rate. default 0.01
    """
    m = X.shape[0]
    n = X.shape[1]
    _theta = theta.copy()  # make a copy to avoid confusion
    cost_data = []
    for _ in range(epoch):
        for i in range(m):
            _theta = _theta - alpha * gradient(_theta, X[i,:].reshape(1, n), [y[i]]) / m
            cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data