import numpy as np
from code_for_lab1 import gen_flipped_lin_separable

def perceptron(data, labels, params={}, hook=None):

    T = params.get('T', 100)

    d, n = data.shape

    th = np.zeros((d, 1))
    th0 = np.zeros((1, 1))

    for t in range(T):

        for i in range(n):

            yi = np.array([labels[[0], [i]]])
            xi = np.array([data.T[i]]).T

            if yi * (th.T @ xi + th0) <= 0:

                th = th + yi * xi
                th0 = th0 + yi

                if hook: hook((th, th0))

    return th, th0


def averaged_perceptron(data, labels, params={}, hook=None):

    T = params.get('T', 100)

    d, n = data.shape

    th, ths = np.zeros((d, 1)), np.zeros((d, 1))
    th0, th0s = np.zeros((1, 1)), np.zeros((1, 1))

    for t in range(T):

        for i in range(n):

            yi = np.array([labels[[0],[i]]])
            xi = np.array([data.T[i]]).T

            if yi * (th.T @ xi + th0) <= 0:

                th = th + yi * xi
                th0 = th0 + yi

                if hook: hook((th, th0))

            ths = ths + th
            th0s = th0s + th0

    return ths / (n * T), th0s / (n * T)


def y(x, th, th0):
    return th.T @ x + th0


def positive(x, th, th0):
    return np.sign(y(x, th, th0))


def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)


def eval_classifier(learner, data_train, labels_train, data_test, labels_test):

    th, th0 = learner(data=data_train, labels=labels_train)

    n_test = data_test.shape[1]

    return score(data_test, labels_test, th, th0) / n_test


def eval_learning_alg(learner, data_gen, n_train, n_test, it):

    scr = 0

    for i in range(it):

        scr += eval_classifier(learner, *data_gen(n_train), *data_gen(n_test))

    return scr / it


def xval_learning_alg(learner, data, labels, k):

    n = data.shape[1]

    data = np.array_split(data, k, axis=1)
    labels = np.array_split(labels, k, axis=1)

    scr = 0

    for j in range(k):

        data_train = np.concatenate([data[i] for i in range(k) if i != j], axis=1)
        labels_train = np.concatenate([labels[i] for i in range(k) if i != j], axis=1)

        data_test = data[j]
        labels_test = labels[j]

        scr += score(data_test, labels_test, *learner(data=data_train, labels=labels_train))

    return scr / n


print(eval_learning_alg(averaged_perceptron, gen_flipped_lin_separable, 20, 20, 100))
