import pandas as pd
import numpy as np
import scipy.optimize as opt

# Load housing data from text file to np array
data = np.loadtxt('ex2data2.txt', delimiter=',')
M = len(data)
# Set column 0-1 as feature and column 2 as target
X = np.array(data[:, 0:2])
Y = np.reshape(np.array(data[:, 2]), (M, 1))  # Reshaped as a column vector


# Original 2 feature function (works)
def mapFeatureOrig(x1, x2, degree):
    out = np.ones((len(x1), 1))
    x1 = np.reshape(x1, (len(x1), 1))
    x2 = np.reshape(x2, (len(x2), 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.concatenate((out, np.reshape(np.power(x1, (i - j)) * np.power(x2, j), (len(x1), 1))), axis=1)
    return out


def mapFeatureOrigTest(x1, x2, degree):
    out = np.ones((len(x1), 1))
    x1 = np.reshape(x1, (len(x1), 1))
    x2 = np.reshape(x2, (len(x2), 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.concatenate((out, np.reshape(x1 ** (i - j) * x2 ** j, (len(x1), 1))), axis=1)
    return out


# Test if changes to the powers makes a difference
X_map = mapFeatureOrig(X[:, 0], X[:, 1], 6)
X_map2 = mapFeatureOrigTest(X[:, 0], X[:, 1], 6)
if np.array_equal(X_map, X_map2):
    test = True
else:
    test = False
print('breakpoint')
# 9-10-20 10:48a changes to powers in line beginning 'out =' do not alter output


# New function for any amount of features (in progress)
def mapFeatureNew(x, degree):
    out = np.ones((len(x), 1))
    for i in range(np.size(x, 1) - 1):
        for j in range(1, degree + 1):
            for k in range(j + 1):
                out = np.concatenate((out, np.reshape(x[:, i] ** (j - k) * x[:, i + 1] ** k, (len(x), 1))), axis=1)
    return out


# Test if new function equals original
X_map3 = mapFeatureNew(X, 6)
if np.array_equal(X_map, X_map3):
    test2 = True
else:
    test2 = False
print('breakpoint')
# 9-10-20 11:16a the j and k in powers were incorrectly labelled as i and j
