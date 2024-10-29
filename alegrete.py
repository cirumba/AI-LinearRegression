import numpy as np


def compute_mse(b, w, data):

    total_error = np.mean((data[:, 1] - (w * data[:, 0] + b)) ** 2)
    return total_error


def step_gradient(b, w, data, alpha):

    N = len(data)
    b_gradient = -2 / N * np.sum(data[:, 1] - (w * data[:, 0] + b))
    w_gradient = -2 / N * np.sum(data[:, 0] * (data[:, 1] - (w * data[:, 0] + b)))
    
    new_b = b - (alpha * b_gradient)
    new_w = w - (alpha * w_gradient)
    
    return new_b, new_w


def fit(data, b, w, alpha, num_iterations):
   
    b_values = [b]
    w_values = [w]

    for i in range(num_iterations):
        b, w = step_gradient(b, w, data, alpha)
        b_values.append(b)
        w_values.append(w)

    return b_values, w_values

