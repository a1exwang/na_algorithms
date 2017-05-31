import math


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

print("x**3-x-1")
newton(
        lambda x: x**3 - x - 1,
        lambda x: 3*x**2 - 1,
        0.6,
        lambda l0, i: l0 * 0.5**i,
        1,
        1e-5,
        lambda k, la, y, y1, x, x1: print("(k, lambda, x, x1, delta_x) = (%d, %0.7f, %0.7f, %0.7f, %0.7f)" %
                                          (k, la, x, x1, abs(x1 - x))))


print("-x**3 + 5*x")
newton(
    lambda x: -x**3 + 5*x,
    lambda x: -3*x**2 + 5,
    1.2,
    lambda l0, i: l0 * 0.5**i,
    1,
    1e-5,
    lambda k, la, y, y1, x, x1: print("(k, lambda, x, x1, delta_x) = (%d, %0.7f, %0.7f, %0.7f, %0.7f)" %
                                      (k, la, x, x1, abs(x1 - x))))
