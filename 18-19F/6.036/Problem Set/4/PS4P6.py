import numpy as np


def gradient_descent(f, df, x_0, eta, n):

    fs, xs = [], []

    for i in range(n - 1):

        fs.append(f(x_0))
        xs.append(x_0.copy())

        x_0 -= eta(i) * df(x_0)

    fs.append(f(x_0))
    xs.append(x_0)

    return x_0, fs, xs


def numerical_gradient(f, delta=0.001):

    def df(x):

        df = np.zeros(np.shape(x))

        for i in range(np.shape(x)[0]):

            delta_i = np.zeros(np.shape(x))
            delta_i[i, :] = delta

            df[i, :] = (f(x + delta_i) - f(x - delta_i)) / (2 * delta)

        return df

    return df


def f1(x):
    return float((2 * x + 3)**2)


def df1(x):
    return 2 * 2 * (2 * x + 3)


def f2(v):
    x, y = float(v[0]), float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y - 1)**2


def df2(v):
    x, y = float(v[0]), float(v[1])
    return np.array([[(-3. + x) * (-2. + x) * (1. + x) + (-3. + x) * (-2. + x) * (3. + x) + (-3. + x) * (1. + x) *
                      (3. + x) + (-2. + x) * (1. + x) * (3. + x) + 2 * (-1. + x + y), 2 * (-1. + x + y)]]).T
