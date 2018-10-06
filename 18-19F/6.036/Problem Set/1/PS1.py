import numpy as np
import math
from random import randint


def length(np_array):

    _len = np.sqrt(np.sum(np.square(np_array), axis=0))

    return _len


def signed_dist(x, theta, theta_0):

    dist = (theta.T @ x + theta_0) / np.array([length(theta)]).T

    return dist


def positive(x, theta, theta_0):

    dist = signed_dist(x, theta, theta_0)
    sign = np.sign(dist)

    return sign


def score(data, labels, theta, theta_0):

    pos = np.array(positive(data, theta, theta_0.T))
    matches = labels == pos
    _sum = np.sum(matches, axis=1)

    return _sum


def best_separator(data, labels, thetas, theta_0s):

    scores = score(data, labels, thetas, theta_0s)
    print(scores)
    max_ind = np.argmax(scores)
    print(max_ind)
    max_th = np.array([thetas.T[max_ind]]).T
    max_th0 = np.array([theta_0s.T[max_ind]]).T

    return max_th, max_th0


def best_separator_loop(data, labels, thetas, theta_0s):

    scores = []

    bst_score = -math.inf
    bst_ind = None
    bst_th = None
    bst_th0 = None

    counter = 0

    for theta, theta_0 in zip(thetas.T, theta_0s.T):

        scr = score(data, labels, theta, theta_0)

        print(scr)

        scores.append(scr)

        if scr > bst_score:

            bst_ind = counter
            bst_score = scr
            bst_th = theta
            bst_th0 = theta_0

        counter += 1

    print(bst_ind)
    return bst_th, bst_th0


D = np.array([[1, 2],
              [1, 3],
              [2, 1],
              [1, -1],
              [2, -1]]).T
L = np.array([[-1, -1, +1, +1, +1]])
THs = np.array([[randint(-5, 5), randint(-5, 5)],
                [randint(-5, 5), randint(-5, 5)],
                [randint(-5, 5), randint(-5, 5)],
                [randint(-5, 5), randint(-5, 5)],
                [randint(-5, 5), randint(-5, 5)],
                [randint(-5, 5), randint(-5, 5)],
                [randint(-5, 5), randint(-5, 5)]]).T
TH0s = np.array([[randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5)]])

print(THs)
print(TH0s)

print(best_separator_loop(D, L, THs, TH0s))
print(best_separator(D, L, THs, TH0s))
