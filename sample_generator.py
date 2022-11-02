import numpy as np


def generate_spiral(n):
    w = np.random.rand(n, 1)
    rho = w * 0.35 + 0.15 + np.random.rand(n, 1) * 0.05
    theta = w * np.pi * 3
    x0 = np.cos(theta) * rho
    x1 = np.sin(theta) * rho
    x = np.concatenate((x0, x1), axis=1)    
    return x