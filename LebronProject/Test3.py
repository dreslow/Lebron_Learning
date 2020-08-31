import pandas as pd
import numpy as np


Data = np.array(pd.read_csv(r'LebronAllStats.csv'))
X_init = np.column_stack((Data[:, :10], Data[:, 19:27]))
np.random.seed(1)
np.random.shuffle(X_init)
y_init = Data[:, 28].reshape((-1, 1)).astype(int)
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




