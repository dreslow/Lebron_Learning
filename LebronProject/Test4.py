import pandas as pd
import numpy as np


Data = np.array(pd.read_csv(r'LebronAllStats.csv'))

# Group samples into mx18 and true results into mx1
X_init = np.column_stack((Data[:, :10], Data[:, 19:27]))
y_init = Data[:, 28].reshape((-1, 1)).astype(int)  # column 27 = pts, column 28 = logistic 10pt buckets
m = len(X_init)

# Randomize order of samples and true results (same seed for same randomization)
np.random.seed(1)
np.random.shuffle(X_init)
np.random.seed(1)
np.random.shuffle(y_init)

# Separate samples into training, cross validation, and test sets
X_train = X_init[:821]
y_train = y_init[:821]
X_CV = X_init[821:1094]
y_CV = y_init[821:1094]
X_test = X_init[1094:]
y_test = y_init[1094:]


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


X_trainNorm, _, _ = featureNormalize(X_train)
X_CVNorm, _, _ = featureNormalize(X_CV)
X_testNorm, _, _ = featureNormalize(X_test)