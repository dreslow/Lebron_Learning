import pandas as pd
import numpy as np
import scipy.optimize as opt

# 9-9-20 4:25 optimization reaches local minima
# 9-9-20 4:34 best guess: data is underfit (high bias) - recommend adding polynomial features
# 9-10-20 9:09 adding polynomial features decreases accuracy in all sets
# Recommend normalizing after adding polynomial features and determining F1 score

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
print('breakpoint')


# Normalize features
def featureNormalize(x):
    x_norm = np.zeros(np.shape(x))
    mu = np.zeros((1, np.size(x, 1)))
    sigma = np.zeros((1, np.size(x, 1)))
    for i in range(np.size(x, 1)):
        mu[0, i] = np.mean(x[:, i])
        sigma[0, i] = np.std(x[:, i])
        x_norm[:, i] = (x[:, i] - mu[0, i]) / sigma[0, i]
    return x_norm


# X_trainNorm = featureNormalize(X_train)
# X_CVNorm = featureNormalize(X_CV)
# X_testNorm = featureNormalize(X_test)
print('breakpoint')


def mapFeature(x, degree):
    out = np.ones((len(x), 1))
    for i in range(np.size(x, 1) - 1):
        for j in range(1, degree + 1):
            for k in range(j + 1):
                out = np.concatenate((out, np.reshape(x[:, i] ** (i - j) * x[:, i + 1] ** j, (len(x), 1))), axis=1)
    return out


# X_trainNormMap = mapFeature(X_trainNorm, 3)
# X_CVNormMap = mapFeature(X_CVNorm, 3)
# X_testNormMap = mapFeature(X_testNorm, 3)

X_trainMap = mapFeature(X_train, 3)
X_CVMap = mapFeature(X_CV, 3)
X_testMap = mapFeature(X_test, 3)
print('breakpoint')
X_trainNormMap = featureNormalize(X_trainMap)
X_CVNormMap = featureNormalize(X_CVMap)
X_testNormMap = featureNormalize(X_testMap)
print('breakpoint')


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Regularized logistic regression cost (Y must be shaped)
def lrCostFunction(theta, x, Y, Lambda):
    M = len(Y)
    theta = theta.reshape(-1, 1)
    hyp = sigmoid(x @ theta)
    theta_c = np.row_stack((np.zeros(1), theta[1:]))
    J = (Y.T @ (np.log(hyp) * -1) - (1 - Y).T @ (np.log(1 - hyp))) / M + Lambda / (2 * M) * np.sum(theta_c ** 2)
    return J


# Regularized logistic regression gradients (Y must be shaped)
def lrGradient(theta, x, Y, Lambda):
    M = len(Y)
    theta = theta.reshape(-1, 1)
    hyp = sigmoid(x @ theta)
    theta_c = np.row_stack((np.zeros(1), theta[1:]))
    return np.ravel((x.T @ (hyp - Y)) / M + Lambda / M * theta_c)


# One vs all learning algorithm (Y must be shaped)
def oneVsAll(x, Y, num_labels, Lambda):
    M = len(x)
    N = x.shape[1]
    x = np.column_stack((np.ones(M), x))
    all_theta = np.random.rand(num_labels, N + 1)
    for c in range(num_labels):
        initial_theta = np.zeros((N + 1, 1))
        theta = opt.fmin_tnc(lrCostFunction, x0=initial_theta, args=(x, Y == c, Lambda), fprime=lrGradient)
        all_theta[c] = theta[0]
    return all_theta


# Parameters for learning
Num_labels = len(np.unique(y_init))
print('breakpoint')
reg_lambda = 1
opt_theta = oneVsAll(X_trainNormMap, y_train, Num_labels, reg_lambda)


# Class predictions based on learned parameters
def predictOneVsAll(all_theta, x):
    M = len(x)
    x = np.column_stack((np.ones(M), x))
    z = all_theta @ x.T
    return np.argmax(sigmoid(z), axis=0).reshape(-1, 1)


# Training set accuracy based on class predictions
pred = np.mean(predictOneVsAll(opt_theta, X_trainNormMap) == y_train) * 100
print('Training set accuracy: ', pred)

# CV set accuracy based on class predictions
pred2 = np.mean(predictOneVsAll(opt_theta, X_CVNormMap) == y_CV) * 100
pred2_results = predictOneVsAll(opt_theta, X_CVNormMap)
print('breakpoint')
print('CV set accuracy: ', pred2)
# 9-10-20 8:59 max CV set accuracy 53.48% w/ lambda = 2

# Test set accuracy based on class predictions
pred3 = np.mean(predictOneVsAll(opt_theta, X_testNormMap) == y_test) * 100
print('Test set accuracy: ', pred3)

# Amount of each result type in the true CV set and predicted CV set
true_result_amounts = np.unique(y_CV, return_counts=True)[1]
pred_result_amounts = np.unique(pred2_results, return_counts=True)[1]
print('breakpoint')
