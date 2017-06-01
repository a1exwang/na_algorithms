from libs import my_sum, newton, iteration_solver, hillbert, power, fitting
import numpy as np


def do_sum():
    print('1.2 Sum')
    my_sum(
        lambda n, sigma: print(n, sigma, 1 / n) if n % 100000 == 0 else None,
        10000000000,
        np.float)


def do_newton():
    print("2.2 Newton")
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


def do_iteration_solver():
    n = 10
    A = hillbert(n)
    b = np.arange(1, n+1, dtype='double')
    x0 = np.zeros(n)
    threshold = 1e-4
    iteration_solver(
        A,
        b,
        x0,
        threshold,
        lambda n, x, delta: print(n, np.linalg.norm(delta), x) if n % 10000 == 0 else None,
        method='sor',
        omega=1.25)

    iteration_solver(
        A,
        b,
        x0,
        threshold,
        lambda n, x, delta: print(n, np.linalg.norm(delta), x) if n % 1 == 0 else None,
        method='jacobi',
        omega=1.25)


def do_power():
    print("Ax = lambda x, what is lambda and x")
    A = np.array([
        [5, -4, 1],
        [-4, 6, -4],
        [1, -4, 7]
    ], dtype='double')
    x0 = np.ones(3, dtype='double')
    power(A, x0, 1e-5,
          lambda i, x, lambda1, delta: print("(i, delta, lambda1, x) = (%d, %f, %f, %s)"
                                             % (i, delta, lambda1, str(x))))
    A = np.array([
        [25, -41, 10, -6],
        [-41, 68, -17, 10],
        [10, -17, 5, -5],
        [-6, 10, -3, 2]
    ], dtype='double')
    x0 = np.ones(4, dtype='double')
    power(A, x0, 1e-5,
          lambda i, x, lambda1, delta: print("(i, delta, lambda1, x) = (%d, %f, %f, %s)"
                                             % (i, delta, lambda1, str(x))))


def do_fitting():
    data = np.array([
        [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
        [33.4, 79.5, 122.65, 159.05, 189.15, 214.15, 238.65, 252.2, 267.55, 280.5, 296.65, 301.65, 310.4, 318.15, 325.15]
    ], dtype='double')
    r, d = fitting(data, 3)
    print("y = %f + %fx + %fx^2, \td = %f" %(r[0], r[1], r[2], d))

    data1 = np.log(data)
    r, d = fitting(data1, 2)
    print("y = %f e^(%ft), \t\t\td = %f" % (np.exp(r[0]), np.exp(r[1]), d))


if __name__ == '__main__':

    do_sum()
    do_newton()
    do_iteration_solver()
    do_power()
    do_fitting()



