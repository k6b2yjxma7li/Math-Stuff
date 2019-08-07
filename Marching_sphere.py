"""
Marching sphere
===

Description:
---
This algorithm is simple approach to using idea of Laplace's mechanics to
deduce extrema of a given function. The file consists of several functions
to perform necessary calculations.
"""


# Assisting functions

def sphere_generator(dim, height, offset):
    """
    Sphere generator
    ---
    Description:
    ---
    Generates matrix containing points layed on a surface of multidimensional
    sphere.

    Parameters:
    ---
    + `dim`: int; specifies order of dimension of a sphere
    + `height`: float; specifies radius of a sphere
    + `offset`: iter(float); specifies center point of a sphere

    Returns:
    ---
    + `result`: list(list(float)) dim x 2dim matrix of points on a sphere

    Raises:
    ---
    + `ValueError`: Offsetting point has incomapatible length -- if `offset`
    length is not equal to `dim` value.
    """
    if len(offset) != dim:
        raise ValueError("Offsetting point has incompatible length:\n"
                         f"dim: {dim}\n"
                         f"len(offset): {len(offset)}")
    if '__len__' not in dir(offset):
        raise TypeError("Offsetting point is not compatible with matrix")
    result = []
    for n in range(dim):
        result_plus = []
        result_minus = []
        for k in range(dim):
            if n == k:
                result_minus.append(-height+offset[n])
                result_plus.append(height+offset[n])
            else:
                result_minus.append(offset[n])
                result_plus.append(offset[n])
        result.append(result_plus+result_minus)
    return result


def marching_sphere(function, start, steps=1e+4, dstnc=1.0, rate=0.5,
                    selector=max):
    """
    Marching sphere
    ---
    Description:
    ---
    By applying idea of Lagrange mechanics it finds extrema of a given
    function.

    Parameters:
    ---
    +`function`: function obj; analysed function
    + `start`: iter(float); starting point of marching sphere
    + `steps`: int; number of marching steps
    + `dist`: float; first step distance (radius of first sphere)
    + `rate`: float; 1/base of exponential distance drop
    + `selector`: function obj; function that selects one of points to
    follow in the next step

    Returns:
    ---
    + `point`: tuple(float); extremum point of `function`

    Raises:
    ---
    """
    rad = dstnc
    dim = len(function.__code__.co_varnames)
    last = start
    data = [() for m in range(dim)]
    try:
        steps = int(steps)
    except ValueError:
        raise ValueError
    for n in range(steps):
        sphere = sphere_generator(dim, rad, last)
        rad *= rate
        if n % 10 == 0:
            rad = dstnc
            dstnc *= rate
        point = [function(*[sphere[l][k] for l in range(dim)])
                 for k in range(2*dim)]
        selected_index = point.index(selector(point))
        last = [sphere[l][selected_index] for l in range(dim)]
        data = [data[m]+(last[m],) for m in range(len(last))]
    return (last, point[selected_index], data)


# Test functions

def main():
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as mpl3d

    def lorentz_3d(A, Bx, By, X, Y):
        return A/(1 + (((2*X)/Bx)**2 + ((2*Y)/By)**2))


    dist_fun = lambda x, y: (lorentz_3d(1, 2, 3, x, y) +
                             lorentz_3d(2, 4, 8, x + 2.1, y + 0.4) -
                             lorentz_3d(1, 2, 3, x - 2, y - 3) +
                             lorentz_3d(6, 0.8, 0.6, x - 2, y - 3))

    L = 41
    x = [10*(n/L)-5 for n in range(L)]
    y = x.copy()
    X = [x[n] for n in range(len(x)) for k in x]
    Y = [y[n] for k in y for n in range(len(y))]
    Z = [dist_fun(xp, yp) for xp in x for yp in y]

    lst, wpoint, dat = marching_sphere(dist_fun, (0, -5), 1000, 2.0, 0.5, max)
    plt.figure()
    plt.axes(projection='3d')
    plt.plot(X, Y, Z, '.', markersize=0.7)
    plt.plot(*dat, [dist_fun(dat[0][m], dat[1][m]) for m in range(len(dat[0]))], '.')

    plt.figure()
    for line in dat:
        plt.plot(range(len(line)), line)
    plt.show()
