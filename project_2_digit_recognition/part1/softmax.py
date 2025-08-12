import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE
    
    # 1. Compute all dot products at once with a single matrix multiplication.
    # The result `scores` is an (n, k) matrix.
    scores = (X @ theta.T) / temp_parameter
    
    # 2. For stability, subtract the max score *for each data point*.
    # We find the max along axis=1 (across the k scores) and use keepdims=True
    # to ensure the result can be broadcast correctly.
    max_scores = np.max(scores, axis=1, keepdims=True)
    stable_scores = scores - max_scores

    # 3. Exponentiate the stable scores.
    exp_scores = np.exp(stable_scores)

    # 4. Sum the exponentiated scores for each data point to get the denominators.
    # We sum along axis=1 and use keepdims=True for the division.
    denominators = np.sum(exp_scores, axis=1, keepdims=True)

    # 5. Compute the probabilities. The result is an (n, k) matrix.
    probabilities = exp_scores / denominators

    # 6. Transpose the result to get the desired (k, n) output shape.
    H = probabilities.T

    return H


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    
    n, d = X.shape
    k, _ = theta.shape
    
    # == Part 1: Cross-Entropy Loss ==
    
    # 1. Calculate scores (logits) for all samples and classes. Shape: (n, k)
    scores = (X @ theta.T) / temp_parameter
    
    # 2. Use the "log-sum-exp" trick for numerical stability.
    # First, find the maximum score for each sample.
    max_scores = np.max(scores, axis=1, keepdims=True)
    
    # Subtract the max score before exponentiating to prevent overflow.
    stable_scores = scores - max_scores
    
    # Calculate the log of the denominators.
    log_sum_exp = max_scores + np.log(np.sum(np.exp(stable_scores), axis=1, keepdims=True))
    
    # Calculate all log probabilities.
    log_probs = scores - log_sum_exp
    
    # 3. Select only the log probabilities corresponding to the true labels (Y).
    # np.arange(n) creates row indices [0, 1, ..., n-1].
    # Y provides the column indices for the correct class for each row.
    correct_log_probs = log_probs[np.arange(n), Y]
    
    # 4. Compute the average negative log-likelihood.
    fit_cost = -np.mean(correct_log_probs)
    
    # == Part 2: Regularization ==
    
    # 5. Calculate the regularization cost.
    reg_cost = (lambda_factor / 2) * np.sum(theta**2)
    
    # == Final Cost ==
    
    # 6. Add the two parts together for the final cost.
    total_cost = fit_cost + reg_cost
    
    return total_cost

            
def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    
    H = compute_probabilities(X, theta, temp_parameter)
    
    n, d = X.shape
    k, _ = theta.shape
    
    data = np.ones(n)
    rows = Y
    cols = np.arange(n)

    # Create the sparse matrix, telling it the final desired shape
    binary_Y_matriz_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(k, n))

    inner_operation = H -binary_Y_matriz_sparse
    matrix_gradient_mult = inner_operation @ X
    loss_gradient = matrix_gradient_mult / (n*temp_parameter)
    regularization_gradient = lambda_factor * theta
    cost_gradient = loss_gradient + regularization_gradient
    new_theta = theta - alpha*cost_gradient
    
    return new_theta.A


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3
    
    return train_y_mod3, test_y_mod3

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    Y_pred = get_classification(X, theta, temp_parameter)
    Y_pred = Y_pred % 3
    
    return 1 - np.mean(Y == Y_pred)

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
