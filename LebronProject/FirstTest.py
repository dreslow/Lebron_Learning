import pandas as pd
import numpy as np
import scipy.optimize as opt


Data = np.array(pd.read_csv(r'LebronCLEstatsTest1.csv'))
X = Data[:, 0:16]
y = np.floor(Data[:, 17]/10).reshape((-1, 1)).astype(int)


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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Neural network cost function (nn_params must be unrolled vector of theta, Y must be shaped)
def nnCostFunction(nn_params, input_layersize, hidden_layersize, num_labels, x, Y, Lambda):
    # Re-rolling parameters
    theta1 = np.reshape(nn_params[:hidden_layersize * (input_layersize + 1)], (hidden_layersize, input_layersize + 1), order='F')
    theta2 = np.reshape(nn_params[hidden_layersize * (input_layersize + 1):], (num_labels, hidden_layersize + 1), order='F')
    M = len(x)

    # Forward propagation
    x = np.column_stack((np.ones(M), x))
    z2 = x @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones(len(a2)), a2))
    a3 = sigmoid(a2 @ theta2.T)

    # Transform individual outputs into logical vectors of length = # of classes
    eye = np.identity(num_labels)
    Y = eye[Y.ravel()].T

    # Cost function
    outer_cost = 0
    for j in range(M):
        outer_cost += np.log(a3[j]) @ -Y[:, j] - np.log(1 - a3[j]) @ (1 - Y[:, j])
    reg_term = Lambda / (2 * M) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))
    return (1 / M) * outer_cost + reg_term


# Neural network gradient function (Repeat - for use in advanced minimization)
def nnGradient(nn_params, input_layersize, hidden_layersize, num_labels, x, Y, Lambda):
    # Re-rolling parameters
    theta1 = np.reshape(nn_params[:hidden_layersize * (input_layersize + 1)], (hidden_layersize, input_layersize + 1), order='F')
    theta2 = np.reshape(nn_params[hidden_layersize * (input_layersize + 1):], (num_labels, hidden_layersize + 1), order='F')
    M = len(x)

    # Forward propagation
    x = np.column_stack((np.ones(M), x))
    z2 = x @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones(len(a2)), a2))
    a3 = sigmoid(a2 @ theta2.T)

    # Transform individual outputs into logical vectors of length = # of classes
    eye = np.identity(num_labels)
    Y = eye[Y.ravel()].T

    # Backpropagation
    delta3 = a3 - Y.T
    delta2 = delta3 @ theta2[:, 1:] * (sigmoid(z2) * (1 - sigmoid(z2)))
    theta1_grad = (delta2.T @ x) / M + Lambda / M * np.column_stack((np.zeros(hidden_layersize), theta1[:, 1:]))
    theta2_grad = (delta3.T @ a2) / M + Lambda / M * np.column_stack((np.zeros(num_labels), theta2[:, 1:]))
    return np.concatenate((np.ravel(theta1_grad, order='F'), np.ravel(theta2_grad, order='F')))


# Randomly initialize parameters for learning
def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init


# Randomly initialize parameters for learning
Input_layersize = X.shape[1]
Hidden_layersize = 25
Num_labels = 6
lambda_t = 3
initial_theta1 = randInitializeWeights(Input_layersize, Hidden_layersize)
initial_theta2 = randInitializeWeights(Hidden_layersize, Num_labels)

# Unroll parameters
initial_NN_params = np.concatenate((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))

# Optimize the cost function
NN_params_int = opt.fmin_tnc(nnCostFunction, x0=initial_NN_params, args=(Input_layersize, Hidden_layersize, Num_labels, X, y, lambda_t), fprime=nnGradient, maxfun=100)
NN_params_opt = NN_params_int[0]
Theta1_opt = np.reshape(NN_params_opt[:Hidden_layersize * (Input_layersize + 1)], (Hidden_layersize, Input_layersize + 1),
                    order='F')
Theta2_opt = np.reshape(NN_params_opt[Hidden_layersize * (Input_layersize + 1):], (Num_labels, Hidden_layersize + 1), order='F')


def predict(theta1, theta2, x):
    M = len(x)
    x = np.column_stack((np.ones(M), x))
    a2 = sigmoid(x @ theta1.T)
    a2 = np.column_stack((np.ones(len(a2)), a2))
    output = sigmoid(a2 @ theta2.T)
    return np.argmax(output, axis=1).reshape(-1, 1)


pred = np.mean(predict(Theta1_opt, Theta2_opt, X) == y) * 100
print('Training set accuracy: ', pred)