import numpy as np
import matplotlib.pyplot as plt
import csv
# import Algo
from scipy.optimize import leastsq
from nano.nano import nearest_val
# from Algo import N_PEAKS
import timeit


def get_inpt_files(path):
    import os
    from glob import glob
    grf_path = os.path.join(path, "grf*.txt")
    hbn_path = os.path.join(path, "hbn*.txt")
    print(grf_path)
    print(hbn_path)
    return (glob(grf_path), glob(hbn_path))


def lorentz(x):
    """
    A*lorentz((x-xm)*hahw)
    """
    return 1/(x**2 + 1)


def gauss(x):
    return 1/((2*np.pi)**0.5) * np.exp(-(x**2)/2)


def smoothing(y, mod=1, q=[]):
    y = np.array(y)

    def single_smoothing(x):
        dx = (x[:-2]+2*x[1:-1]+x[2:])/4
        left = [(x[0]+x[1])/2]
        right = [(x[-2]+x[-1])/2]
        return np.append(left, np.append(dx, right))

    for n in range(mod):
        y = single_smoothing(y)
    return y


def d(u):
    u = np.array(u)
    centre = list((u[2:]-u[:-2])/4)
    left_margin = [(u[1]-u[0])/2]
    right_margin = [(u[-1]-u[-2])/2]
    return np.array(left_margin + centre + right_margin)


def gauss_av(sgm, x, y):
    res = []
    G = lambda s: lambda x: 1/(s*(2*np.pi)**(0.5)) * np.exp(-(x**2)/(2*s**2))
    gs = G(sgm)
    for n in range(len(x)):
        g = gs(x-x[n])
        res += [sum(g*y*d(x))/(sum(g*d(x)))]
    return np.array(res)


def div2(function, dx):
    def DIV2(x):
        return (function(x+dx) - 2*function(x) + function(x-dx))/(dx**2)
    return DIV2


def norm(V):
    V = np.array(V)
    return (V.dot(V))**0.5


def grad(function, dv):
    """
    Iterating through all parameters
    """
    def GRAD(vec):
        vec = np.array(vec)
        grad_v = []
        for n in range(len(vec)):
            if "__len__" not in dir(dv):
                dx = len(vec)*[dv]
            else:
                dx = dv.copy()
            dx_n = np.array([0]*n + [dx[n]] + [0]*(len(dx) - 1 - n))
            df = (function(vec+dx_n)-function(vec-dx_n))
            dt = 2*dx[n]
            grad_v.append(df/dt)
        return np.array(grad_v)
    return GRAD


def grad2(function, dv):
    def GRAD2(vec):
        vec = np.array(vec)
        grad_v = []
        for n in range(len(vec)):
            if "__len__" not in dir(dv):
                dx = len(vec)*[dv]
            else:
                dx = dv.copy()
            dx_n = np.array([0]*n + [dx[n]] + [0]*(len(dx) - 1 - n))
            grad_v.append((function(vec+dx_n)-2*function(vec) +
                           function(vec-dx_n))/(dx[n]**2))
        return np.array(grad_v)
    return GRAD2


def newton(function, dt, velocity=1):
    """
    Newton method of finding zeros of function.
    ---
    Newton's algorithm of finding the root of a given function
    works as follows:
    1. derivative `f` of function `F` at point `x0` is calculated;

    2. linear function crossing the point [x0, F(x0)] is defined:
    > `a = f(x0)`\\
    > `a*x0 + b = F(x0)`\\
    > `b = F(x0) - a*x0`\\
    > `y(t) = a*t + b = f(x0)*t + F(x0) - f(x0)*x0`\\
    > `y(t) = f(x0)*(t - x0) + F(x0)`

    3. root of the linear function is calculated:
    > `y(t) = f(x0)*(t - x0) + F(x0) = 0`\\
    > `f(x0)*(t - x0) = -F(x0)`\\
    > `t - x0 = -F(x0)/f(x0)`\\
    > `t = x0 - F(x0)/f(x0)`

    4. point `t` is a new `x0`, back to the point 1.

    This approach can be applied for N-dim functions as well;
    however some differences may be pointed out. First of all
    for N-dim case derivative is calculated as vector derivative
    over a unit vector `n` which in fact is following the direction
    of gradient. Overall equation of resulting vector is:
    > `v = -(F(v0)*grad(F)(v0))/(norm(grad(F)(v0))**2) + v0`

    and this is to be the next point of algorithm.
    """
    def norm(V):
        V = np.array(V)
        return (V.dot(V))**0.5

    def NEWT(v0):
        v0 = np.array(v0)
        F = function
        abs_v0 = norm(v0)
        if abs_v0 != 0:
            n = v0/norm(v0)   # n vec for grad
        else:
            n = np.array(len(v0)*[0])
        grad_F = grad(function, n*dt)
        grad_F_v0 = grad_F(v0)
        F_v0 = F(v0)
        len_grd_F_v0 = norm(grad_F_v0)**2
        # x = grad_F_v0[0]
        # y = grad_F_v0[1]
        # theta = np.arctan(-x/y)
        # r = random.random()
        return (-(F_v0*grad_F_v0)/((len_grd_F_v0+1e-15)))*velocity + v0

    return NEWT


def spectrum(x, v, bias=[0, 0, 0]):
    x = np.array(x)
    result = np.zeros(len(x))
    for n in range(0, len(v), 3):
        A = v[n] + bias[0]
        b = v[n+1] + bias[1]
        x0 = v[n+2] + bias[2]
        result += np.array(A*lorentz(1/b * (x-x0)))
    return result


def err(func, x, y):
    x = np.array(x)
    y = np.array(y)

    def fit_fun(V):
        vec = y - func(x, V)
        return vec

    return fit_fun


def density(t, u, sgm):
    res = np.zeros(len(t))
    for u0 in u:
        res += 1/sgm * gauss((t-u0)/sgm)/len(u)
    return res

# if __name__ == "__main__":
#     main()
