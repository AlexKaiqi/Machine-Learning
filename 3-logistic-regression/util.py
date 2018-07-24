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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
    
def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))    
    
    
def gradient(theta, X, y):
    '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    
    
def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)
    
def regularized_cost(theta, X, y, lam=1):
    '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (lam / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term

def regularized_gradient(theta, X, y, lam=1):
    '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (lam / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term
    