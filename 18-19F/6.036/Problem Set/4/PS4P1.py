import numpy as np


def length(x):
    return np.sqrt(np.sum(np.square(x), axis=0))


def margin(x, y, theta, theta_0):
    return (y * (theta.T @ x + theta_0)) / length(theta)


def score_sum(xs, ys, theta, theta_0):
    return np.sum(margin(xs, ys, theta, theta_0))


def score_min(xs, ys, theta, theta_0):
    return np.min(margin(xs, ys, theta, theta_0))


def score_max(xs, ys, theta, theta_0):
    return np.max(margin(xs, ys, theta, theta_0))


data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,    2,    2,    2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
theta = np.array([[1],
                  [0]])
theta_0 = -2.5

mg = margin(data, labels, theta, theta_0)
sm = score_sum(data, labels, theta, theta_0)
mn = score_min(data, labels, theta, theta_0)
mx = score_max(data, labels, theta, theta_0)

pass
