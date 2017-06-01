import math
import numpy as np


def hillbert(n):
    ret = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            ret[i][j] = 1.0 / (i + j + 1)

    return ret


def my_sum(cb, max_count, dt):
    sigma = dt(0.0)
    n = dt(1.0)
    while True:
        sigma1 = sigma + dt(1) / n
        cb(n, sigma)
        if sigma1 == sigma:
            break
        if n == max_count:
            break
        n += dt(1)
        sigma = sigma1
    return sigma


def newton(f, df, x0, get_lambda, lambda0, epsilon, cb):
    k = 0
    x = x0
    while True:
        s = f(x) / df(x)
        x1 = x - s
        i = 0
        y = f(x)
        la = 1
        while True:
            y1 = f(x1)
            if abs(y1) < abs(y):
                break
            la = get_lambda(lambda0, i)
            x1 = x - la * s
            i += 1
        cb(k, la, y, y1, x, x1)
        k += 1
        if abs(x1 - x) <= epsilon:
            break
        x = x1


def iteration_solver(A, b, x0, threshold, cb, method='jacobi', omega=1):
    x = x0
    iter_count = 0
    n = list(np.shape(b))[0]
    if method == 'jacobi':
        while True:
            y = np.array(x)
            for i in range(n):
                a = np.dot(A[i], x) - A[i, i] * x[i]
                new_xi = (b[i] - a) / A[i, i]
                y[i] = new_xi
            if np.linalg.norm(y - x) <= threshold:
                break
            x = y
            cb(iter_count, x, y - x)
            iter_count += 1
        return x
    else:
        while True:
            y = np.array(x)
            for i in range(n):
                a = np.dot(A[i], x) - A[i, i] * x[i]
                new_xi = (1 - omega) * x[i] + omega * (b[i] - a) / A[i, i]
                x[i] = new_xi
            if np.linalg.norm(y - x) <= threshold:
                break
            cb(iter_count, x, y - x)
            iter_count += 1
        return x


def power(A, x0, threshold, cb):
    u = x0
    last_lambda1 = 0
    i = 0
    while True:
        v = np.dot(A, u)
        lambda1 = np.max(v)
        u = v / lambda1
        delta = abs(last_lambda1 - lambda1)
        if delta < threshold:
            break
        cb(i, u, lambda1, delta)
        last_lambda1 = lambda1
        i += 1
    return lambda1, u


def fitting(pts, n):
    phi = []
    for i in range(n):
        phi.append(np.power(pts[0], i))

    A = np.zeros((n, n), dtype='double')
    for i in range(n):
        for j in range(n):
            A[i, j] = np.dot(phi[i], phi[j])

    b = np.zeros(n, dtype='double')
    for i in range(n):
        b[i] = np.dot(phi[i], pts[1])

    x = np.dot(np.linalg.inv(A), b)

    tt = 0
    for _ in range(list(np.shape(pts))[1]):
        A = np.zeros((n, n), dtype='double')
        for i in range(n):
            for j in range(n):
                A[i, j] = pts[0][i] * pts[0][j]
        tt += np.linalg.norm(np.dot(A, x) - b) ** 2
    d = np.sqrt(tt / list(np.shape(pts))[1])
    return x, d

