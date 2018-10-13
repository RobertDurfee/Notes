import PS4P1 as p1
import numpy as np
import math


def hinge_loss(x, y, theta, theta_0, gamma_ref):
    return (1 - (p1.margin(x, y, theta, theta_0) / gamma_ref)).clip(min=0)


data = np.array([[1.1, 1, 4],
                 [3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
theta = np.array([[1],
                  [1]])
theta_0 = -4

hl = hinge_loss(data, labels, theta, theta_0, math.sqrt(2) / 2)
