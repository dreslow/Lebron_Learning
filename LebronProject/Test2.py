import pandas as pd
import numpy as np
import scipy.optimize as opt

Data = np.array(pd.read_csv(r'LebronCLEstatsTest1.csv'))
X = Data[:500, :16]
y = Data[:500, 17].reshape((-1, 1))
m = len(X)


# Normalize features
def featureNormalize(x):
    x_norm = x
    mu = np.zeros((1, np.size(x, 1)))
    sigma = np.zeros((1, np.size(x, 1)))
    for i in range(np.size(x, 1)):
        mu[0, i] = np.mean(x[:, i])
        sigma[0, i] = np.std(x[:, i])
        x_norm[:, i] = (x[:, i] - mu[0, i]) / sigma[0, i]
    return x_norm, mu, sigma


X_norm, Mu, Sigma = featureNormalize(X)
X_norm = np.column_stack((np.ones(m), X_norm))
theta_initial = np.zeros((X_norm.shape[1]), dtype=float)


def costFunctionLinear(x, Y, theta, Lambda):
    M = len(Y)
    predictions = x @ theta
    sqrErrors = (predictions - Y) ** 2
    return 1 / (2 * M) * np.sum(sqrErrors)  # + Lambda * np.sum(theta ** 2)


theta_opt = np.linalg.inv(X_norm.T @ X_norm) @ X_norm.T @ y
print('Training set cost: ', costFunctionLinear(X_norm, y, theta_opt, 1))
