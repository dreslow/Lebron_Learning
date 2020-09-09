import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


Data = np.array(pd.read_csv(r'LebronAllStats.csv'))
X_init = np.column_stack((Data[:, :10], Data[:, 19:27]))
np.random.seed(1)
np.random.shuffle(X_init)
y_init = Data[:, 27].reshape((-1, 1)).astype(int)
np.random.seed(1)
np.random.shuffle(y_init)
X = X_init[:1000]
y = y_init[:1000]
X_test = X_init[1000:]
y_test = y_init[1000:]
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


X_norm, _, _ = featureNormalize(X)


def mapFeature(x, degree):
    out = np.ones((len(x), 1))
    for i in range(np.size(x, 1) - 1):
        for j in range(1, degree + 1):
            for k in range(j + 1):
                out = np.concatenate((out, np.reshape(x[:, i] ** (i - j) * x[:, i + 1] ** j, (len(x), 1))), axis=1)
    return out


X_mapped = mapFeature(X_norm, 2)


def costFunctionLinear(theta, x, Y, Lambda):
    M = len(Y)
    predictions = x @ theta
    sqrErrors = (predictions - Y) ** 2
    theta_c = np.row_stack((np.zeros(1), theta[1:].reshape(-1, 1)))
    return 1 / (2 * M) * np.sum(sqrErrors) + Lambda * np.sum(theta_c ** 2)


def gradientLinear(theta, x, Y, Lambda):
    M = len(Y)
    predictions = (x @ theta).reshape(-1, 1)
    theta_c = np.row_stack((np.zeros(1), theta[1:].reshape(-1, 1)))
    return np.ravel((x.T @ (predictions - y)) / M + Lambda / M * theta_c)


def gradientDescent(theta, x, Y, Lambda, alpha, iterations):
    J_history = np.zeros((iterations, 1))
    for i in range(iterations):
        delta = gradientLinear(theta, x, Y, Lambda)
        theta = theta - alpha * delta
        J_history[i] = costFunctionLinear(theta, x, Y, Lambda)
    return theta, J_history


initial_theta = np.zeros(len(X_mapped[0]))
Alpha = 0.1
Iterations = 500
Lambda_t = 1
theta_result, J_History = gradientDescent(initial_theta, X_mapped, y, Lambda_t, Alpha, Iterations)

plt.plot(list(range(0, Iterations)), J_History, linewidth=1, color='blue')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# test = gradientLinear(initial_theta, X_mapped, y, 1)
# theta_opt = opt.fmin_tnc(costFunctionLinear, x0=initial_theta, args=(X_mapped, y, 1), fprime=gradientLinear)
# predictions_final = X_mapped @ theta_opt[0]
# print('test')
