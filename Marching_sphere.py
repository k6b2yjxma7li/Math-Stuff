"""
Marching sphere
===
Author: k6b2yjxma7li

Description:
---
This algorithm is simple approach to using idea of Lagrange mechanics to
deduce extrema of a given function. The file consists of several functions
to perform necessary calculations. Caution: it takes a lot of time.
"""
import logging

# LOGGING BLOCK

LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"    # log format
FILE_NAME = __file__.split('/')[-1]     # name of file to include in log
LOG_NOTE = ""                           # special variable for sidenotes

# quieting annoying matplotlib DEBUG messages (a lot of them)
# logging.getLogger('matplotlib').setLevel(logging.WARNING)

while True:
    # check for existence of logging directory
    try:
        logging.basicConfig(filename="./__logs__/main.log",
                            level=logging.DEBUG,
                            format=LOG_FORMAT)
        if LOG_NOTE:
            logging.info(LOG_NOTE)
        logging.info(f"Logging started for `{FILE_NAME}`.")
        break
    except FileNotFoundError:
        import os
        os.mkdir("./__logs__")
        LOG_NOTE = f"{FILE_NAME}: Logging directory created."

# END OF LOGGING BLOCK


# Assisting functions

def sphere_generator(dim, height, offset):
    """
    `Marching sphere` function: `sphere_generator`
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
    # logging.debug("sphere_generator"
    #               f"{tuple([type(v) for k, v in locals().items()])}")
    # logging
    if '__len__' not in dir(offset):
        msg = ("sphere_generator: Offsetting point is not compatible with"
               " matrix.")
        logging.error(msg)
        raise TypeError(msg)
    if len(offset) != dim:
        msg = ("sphere_generator: Offsetting point has incompatible length: "
               f"dim: {dim} != len(offset): {len(offset)}.")
        logging.error(msg)
        raise ValueError(msg)
    # defining return of this function
    result = []
    # height might be defined as a single value
    # OR
    # as a set of values, but this way it must correspond to the dimension
    if '__len__' not in dir(height):
        height = dim*[height]
    elif len(height) != dim:
        msg = ("sphere_generator: Height data have incompatible length: "
               f"dim: {dim} != len(offset): {len(offset)}.")
        logging.error(msg)
        raise ValueError(msg)
    # defining actual sphere
    for n in range(dim):
        # sphere as a matrix contains two parts:
        result_plus = []
        result_minus = []
        # because the structure presents as follows:
        # [[1, 0, 0, ..., -1,  0,  0, ...],
        #  [0, 1, 0, ...,  0, -1,  0, ...],
        #  [0, 0, 1, ...,  0,  0, -1, ...],
        #               ...                ]
        for k in range(dim):
            # adding elements with offset point
            if n == k:
                result_minus.append(-height[n]+offset[n])
                result_plus.append(height[n]+offset[n])
            else:
                result_minus.append(offset[n])
                result_plus.append(offset[n])
        # final merging of results
        result.append(result_plus+result_minus)
    return result


def marching_sphere(function, start, steps=1e+4, dstnc=1.0, rate=0.5,
                    selector=max):
    """
    `Marching sphere` function: `marching_sphere`
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
    + AttributeError: Argument detection failed in function -- if passed
    function has `*args` as one of arguments or does not own special attribute
    `argcount`.
    + ValueError: Argument `steps` of type `{type(steps)}` is not a valid
    numerical value -- if `steps` cannot be converted to `int`.
    + TypeError: Start {start} and distance {dstnc} are not compatible.
    """
    logging.debug("marching_sphere"
                  f"{tuple([type(v) for k, v in locals().items()])}")
    if '__len__' not in dir(dstnc):
        dstnc = len(start)*[dstnc]
    elif len(dstnc) != len(start):
        msg = (f"marching_sphere: Start {start} and distance {dstnc}"
               " are not compatible.")
        logging.error(msg)
        raise TypeError(msg)
    rad = dstnc
    varnames = function.__code__.co_varnames
    # function has to have:
    # special attribute `argcount`
    # OR
    # literal arguments (not *args/**kwargs)
    # one of those two specifies number of regulated parameters
    try:
        dim = function.argcount
    except AttributeError:
        if 'args' not in varnames:
            dim = len(varnames)
        else:
            msg = ("marching_sphere: Argument detection failed in"
                   f" function `{function.__name__}`.")
            logging.error(msg)
    # argument switch to apply recursion; output becomes input
    last = start
    data = [(start[m],) for m in range(dim)]
    try:
        steps = int(steps)
    except ValueError:
        msg = (f"marching_sphere: Argument `steps` of type `{type(steps)}`"
               " is not a valid numerical value.")
        logging.error(msg)
        raise ValueError(msg)
    progress = 100
    the_progress = 0
    final_countdown = 0
    for n in range(steps):
        try:
            # set of points that are located on axes of multidimensional space
            sphere = sphere_generator(dim, rad, last)
            # this was first attempt at regulating accuracy; now obsolete
            rad = list(map(lambda x: x*rate, rad))
            # accuracy change step
            if progress < 1:
                rate = rate**rate
                rad = dstnc
                dstnc = list(map(lambda x: x*rate, dstnc))
            if progress == 0:
                final_countdown += 1
            else:
                final_countdown = 0
            if final_countdown > 7:
                msg = ("marching_sphere: Premature loop break due"
                       " to no progress.")
                logging.warn(msg)
                print(msg)
                break
            # list of points from which one is selected
            point = [function(*[sphere[l][k] for l in range(dim)])
                     for k in range(2*dim)]
            # error of wrong input function result
            if type(point[0]) not in [float, int]:
                msg = (f"marching_sphere: Function `{function.__name__}` does"
                       " not return single numerical value as output.")
                logging.error(msg)
                raise TypeError(msg)
            # id of selected point of a sphere
            selected_index = point.index(selector(point))
            # actual selected point (basing on id)
            last = [sphere[l][selected_index] for l in range(dim)]
            progress = (1-function(*last)/function(*[d[-1] for d in data]))*100
            the_progress = (1-function(*last)/function(*start))*100
            print(f"{n}\t{progress}%\t{the_progress}%")
            # saving all points to recreate changes, not necessary
            data = [data[m]+(last[m],) for m in range(len(last))]
        except KeyboardInterrupt:
            msg = "marching_sphere: User interruption."
            logging.info(msg)
            print(msg)
            break
    return (last, point[selected_index], data)


# Test functions

# def main():
    # """
    # `Marching sphere` function: `main`
    # ---
    # """
#     import matplotlib.pyplot as plt
#     import mpl_toolkits.mplot3d as mpl3d

#     def lorentz_3d(A, Bx, By, X, Y):
#         return A/(1 + (((2*X)/Bx)**2 + ((2*Y)/By)**2))


#     dist_fun = lambda x, y: (lorentz_3d(1, 2, 3, x, y) +
#                              lorentz_3d(2, 4, 8, x + 2.1, y + 0.4) -
#                              lorentz_3d(1, 2, 3, x - 2, y - 3) +
#                              lorentz_3d(6, 0.8, 0.6, x - 2, y - 3))

#     L = 41
#     x = [10*(n/L)-5 for n in range(L)]
#     y = x.copy()
#     X = [x[n] for n in range(len(x)) for k in x]
#     Y = [y[n] for k in y for n in range(len(y))]
#     Z = [dist_fun(xp, yp) for xp in x for yp in y]

#     lst, wpoint, dat = marching_sphere(dist_fun, (0, -5),
#                                        1000, 2.0, 0.5, max)
#     plt.figure()
#     plt.axes(projection='3d')
#     plt.plot(X, Y, Z, '.', markersize=0.7)
#     plt.plot(*dat, [dist_fun(dat[0][m], dat[1][m])
#                     for m in range(len(dat[0]))], '.')

#     plt.figure()
#     for line in dat:
#         plt.plot(range(len(line)), line)
#     plt.show()
