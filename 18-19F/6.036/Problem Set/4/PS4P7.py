import numpy as np
import PS4P6 as p6


def length(x):
    return np.sqrt(np.sum(np.square(x), axis=0))


def hinge(v):
    return np.where(v < 1, 1 - v, 0)


def hinge_loss(x, y, theta, theta_0):
    return hinge(y * (theta.T @ x + theta_0))


def svm_objective(x, y, theta, theta_0, lambda_):
    return np.mean(hinge_loss(x, y, theta, theta_0)) + lambda_ * theta.T @ theta


def d_hinge(v):
    return np.where(v >= 1, 0, -1)


def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T, x) + th0)) * y * x


def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T, x) + th0)) * y


def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0), axis=1, keepdims=True) + lam * 2 * th


def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0), axis=1, keepdims=True)


def svm_obj_grad(x, y, th, th0, lam):
    grad_th = d_svm_obj_th(x, y, th, th0, lam)
    grad_th0 = d_svm_obj_th0(x, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])


def batch_svm_min(data, labels, lam):

    d, n = np.shape(data)

    def svm_min_step_size_fn(i):
        return 2 / (i + 1) ** 0.5

    def pack(th, th0):
        return np.concatenate((th, th0))

    def unpack(x):
        return x[0:d], x[d:d + 1]

    return p6.gradient_descent(f=lambda x: svm_objective(data, labels, *unpack(x), lam),
                               df=lambda x: svm_obj_grad(data, labels, *unpack(x), lam),
                               x_0=pack(np.zeros((d, 1)), np.zeros((1, 1))),
                               eta=svm_min_step_size_fn,
                               n=10)


data = np.array([[2, 3, 9, 12],
                 [5, 2, 6, 5]])
labels = np.array([[1, -1, 1, -1]])
lambda_ = 0.0001

res = batch_svm_min(data, labels, lambda_)

data = np.array([[2, -1, 1, 1],
                 [-2, 2, 2, -1]])
labels = np.array([[1, -1, 1, -1]])
lambda_ = 0.0001

res = batch_svm_min(data, labels, lambda_)

pass
