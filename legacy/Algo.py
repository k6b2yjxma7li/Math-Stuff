"""
Algo
===
Author: k6b2yjxma7li
version: 200819

Description:
---
This is main algorithm file for finding fit of series of Lorentz's curves
to a given spectral data, especially Raman.

Following script had been written by rules of pylama code audit (W291)
Python 3.6.8 (64-bit) and with Visual Studio Code, therefore it gives no
errors nor warnings with this setup.
"""

import matplotlib.pyplot as plt     # plots!
import DataStruct as ds             # to use some of functions
from Marching_sphere import marching_sphere   # fitting!
import math
import time                         # estimating elapsed time
import logging                      # logging!

exp = math.exp
log = math.log

global N_PEAKS
N_PEAKS = 10

# LOGGING BLOCK

LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"    # log format
FILE_NAME = __file__.split('/')[-1]     # name of file to include in log
LOG_NOTE = ""                           # special variable for sidenotes

# quieting annoying matplotlib DEBUG messages (a lot of them)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
# ALGORITHM BLOCK


def lorentz(A, B, x0, x_arg):
    """
    `Algo` function: `lorentz`
    ---

    Description:
    ---
    Lorentz's function generator. Uses three paramters `A`, `B`, `x0` to
    return value of Lorentz's function at point specified in `x_arg`.

    Value of `B` represents half-maximum peak width of Lorentz curve.

    Parameters:
    ---
    + `A` -- peak's amplitude
    + `B` -- half-maximum peak width
    + `x0` -- argument of maximum point
    + `x_arg` -- curve's argument
    """
    return abs(A)/((2*(x_arg-x0)/B)**2.0 + 1.0)


def lorentz2(A, b, x0, x_arg):
    """
    `Algo` function: `lorentz`
    ---

    Description:
    ---
    Lorentz's function generator. Uses three paramters `A`, `b`, `x0` to
    return value of Lorentz's function at point specified in `x_arg`.

    Value of `b` represents coefficient of curve's width as in quadratic
    function.

    Parameters:
    ---
    + `A` -- peak's amplitude
    + `b` -- width coefficient
    + `x0` -- argument of maximum point
    + `x_arg` -- curve's argument
    """
    return abs(A)/(1+(b*(x_arg-x0))**2)


def lorentz3(A, C, x0, x_arg):
    """
    `Algo` function: `lorentz`
    ---

    Description:
    ---
    Lorentz's function generator. Uses three paramters `A`, `C`, `x0` to
    return value of Lorentz's function at point specified in `x_arg`.

    Value of `C` represents coefficient of curve's shape.

    Parameters:
    ---
    + `A` -- peak's amplitude
    + `C` -- shape coefficient
    + `x0` -- argument of maximum point
    + `x_arg` -- curve's argument
    """
    return abs(A)/(abs(C)+(x_arg-x0)**2)


def array_it(function):
    """
    `Algo` function: `array_it`
    ---

    Description:
    ---
    Creating element-wise function object, which then might be used on
    iterator argument.

    Parameters:
    ---
    + `function` -- non-iterating function object
    """
    logging.debug(f"array_it({type(function)}: {function.__name__})")

    def release(P, Q):
        P.update(Q)
        return P

    def list_fun(*args, **kwargs):
        args = list(args)
        try:
            varnames = function.__code__.co_varnames
        except AttributeError:
            raise AttributeError
        for n in range(len(args)):
            if ('__len__' in dir(args[n])) and (type(args[n]) is not str):
                key_lt = varnames[n]
                x_list = args[n]
            else:
                kwargs = release(kwargs, {varnames[n]: args[n]})
        return [function(**(release(kwargs, {key_lt: x}))) for x in x_list]
    return list_fun


lorentz_arr = array_it(lorentz)  # creating arrayed function of lorentz


def error_gen(some_fun, y_data, x_data, start):
    x = x_data
    y = y_data

    def sq_err(*args):
        spectrum = [sum([some_fun(*args[3*m:3*m+3], x[n])
                         for m in range(int(len(args)/3))])
                    for n in range(len(x))]
        return sum([(spectrum[n] - y[n])**2 for n in range(len(y))])
    # special attribute for functions with hard to detect length arguments
    sq_err.argcount = len(start)
    return sq_err


def peak_find(y_arg):
    """
    `Algo` function: `peak_find`
    ---

    Description:
    ---
    Finds peak's parameters basing on `y_arg` argument containing spectral
    data.

    Returns:
    ---
    three-element tuple:
    + 0: function wrapper for Lorentz with args: `b`, `x`, `x0`
    + 1: index of peak maximum in `y_arg`
    + 2: amplitude of peak
    """
    logging.debug(f"peak_find({type(y_arg)})")
    data = y_arg
    data_c = y_arg.copy()
    data_c.reverse()
    minimum = ds.Helper.gravity_mean(data_c)    # min to use as background
    amp = max(data)-minimum               # amplitude
    main_point = ds.nearest(data, max(data))    # finding peak point arg
    return (lambda b, x, x0: lorentz_arr(amp, b, x0, x),
            main_point, amp)


"""
In the following section there are definitions for fine tuning of half-amp
peak width, aka `b` value.
"""

X = [0.01*n for n in range(-1000, 1000)]        # data range [-10;10]
Y = [1-math.exp(-0.1*x**2) for x in X]          # flipped Gauss curve from X
EY = [0.01*sum(Y[:n])-0.01*sum(Y)/2 for n in range(len(Y))]  # set to check

"""
Following method is lead function, that calculates parameters of Lorentz's
functions to estimate given spectrum.
"""


def observe(y_arg, x_arg, peaks_nr=N_PEAKS, bias=[0, 0, 0]):
    """
    `Algo` function: `observe`
    ---

    Description:
    ---
    Swipes through data, finds peaks by using `peak_find` method, and matches
    certain parameters of each Lorentz to finally return n-row 3-element list
    of parameters in the following order: [amplitude, half-amp width, peak arg]

    Returns:
    ---
    + `data_set`: list(list(float, float, float)) -- list of vectors:
    [`amplitude`, `half-amp width`, `peak arg`]
    """
    logging.debug(f"observe({type(y_arg)}, {type(x_arg)})")
    sum_time = 0.0
    data_set = []
    for akn in range(peaks_nr):                     # find N_PEAKS nr of curves
        start = time.process_time()
        peaks = peak_find(y_arg)                     # peak finding
        b_val = 0
        for m in range(1, 10000, 10):               # course-search for-loop
            s = peaks[0](m+bias[1], x_arg, x_arg[peaks[1]]+bias[2])     # lorentz_arr
            ch_val = ds.pearson(y_arg, s)           # Pearson's R value check
            if ch_val > b_val:
                b_val = ch_val
            else:
                b_val = m
                break
        fine_val = 0
        for m in range(len(EY)):                    # fine-search for-loop
            s = peaks[0](b_val+EY[m]+bias[1], x_arg, x_arg[peaks[1]]+bias[2])
            ch_val = ds.pearson(y_arg, s)
            if ch_val > fine_val:
                fine_val = ch_val
            else:
                fine_val = b_val+EY[m]+bias[1]
                break
        data_set.append([peaks[2]+bias[0], fine_val, x_arg[peaks[1]]+bias[2]])  # main dataset
        # data_set.append(out[0])  # main dataset
        lor = lorentz_arr(peaks[2]+bias[0], fine_val, x_arg, x_arg[peaks[1]]+bias[2])
        # lor = lorentz_arr(out[0][0], out[0][1], x_arg, out[0][2])
        # y_arg change to eliminate found peaks
        y_arg = [y_arg[n]-lor[n] for n in range(len(y_arg))]
        logging.info(f"observe: step {akn}: {time.process_time()-start}\ts")
        sum_time += time.process_time()-start
    logging.info(f"observe: Time consumption: {sum_time}.")
    return data_set


def main(x_data, y_data, scales=N_PEAKS*[1, 1, 1],
         function=lambda a, b, x0, x: lorentz2(a, b, x0, x),
         steps=200, rate=0.5, selector=min, graphic=False):
    """
    `Algo` main method
    ---

    Description:
    ---
    Uses `Algo.observe` method to fit Lorentz's curves to data, then
    generates fit points from return of `observe`. After that the function
    improves fitting with `Marching_sphere.marching_sphere`. For all steps it
    calculates time needed to complete the task. Finally it presents
    results in form of a plot with three lines:
        + blue -- actual data
        + orange -- fit
        + green -- absolute error

    Parameters:
    ---
    + `waveform_nr`:int -- defines number of a waveform from __Main file
    + `scales`:list(float) -- defines size of step ranges reffering to
    `observe` fit values
    + `function`:function_obj -- fitted peak function
    + `steps`:int -- number of `marching_sphere` steps
    + `rate`:float -- scaling value of `marching_sphere`
    + `selector`:function_obj -- `marching_sphere` selecting function
    + `graphic`:bool -- plots showed immediately if `True`
    For further explanation of least three parameters see docstring of
    `Marching_sphere.marching_sphere`.

    Warning!
    ---
    This version does not support actual changes in peak function at the
    moment. Following updates will introduce fully custom fit functions.
    Work is in progress.
    """
    global N_PEAKS
    start = time.process_time()
    # Marching sphere block
    # expected to take 0.5s/iteration
    print("Observing started...")
    fit = observe(y_data, x_data, N_PEAKS)   # main algorithm call
    print(f"Spectrum observed. Time: {time.process_time()-start}s")
    fit = [[abs(f[0]), abs(2/f[1]), f[2]] for f in fit]
    fit_tmp = []
    [fit_tmp.extend(f) for f in fit]
    fit = fit_tmp
    # scales = N_PEAKS*[0.9, 5.0, 2.0]
    step = [fit[n]*scales[n] for n in range(len(fit))]
    err_fun = error_gen(function, y_data, x_data, fit)
    logging.debug(f"Starting point: {fit}")
    print(f"Starting error:\t{err_fun(fit)}")
    new_fit = marching_sphere(function=err_fun,
                              start=fit,
                              steps=steps,
                              dstnc=step,
                              rate=rate,
                              selector=selector)
    marching_stop = time.process_time()
    msg = f"main: Marching time: {marching_stop-start}."
    m_fit = [new_fit[0][3*m:3*(m+1)] for m in range(N_PEAKS)]
    print(msg)
    logging.info(msg)
    w_len = len(x_data)
    lor = [0 for n in range(w_len)]   # init of estimation dataset
    print(f"Final error: {err_fun(new_fit[0])}")
    logging.debug(f"Ending point: {new_fit[0]}")
    # this gonna take some time, but also it will count it!
    sum_time = 0.0
    main_time = time.process_time()
    msg = "main: Fit spectrum generation."
    logging.info(msg)
    print(msg)
    for f in range(len(m_fit)):      # for-loop to sum all Lorentz's
        step = time.process_time()
        for n in range(w_len):
            lor[n] += function(*m_fit[f], x_data[n])
        stop = time.process_time()
        sum_time += stop-step
        logging.info("main: Approximation ETA:"
                     f" {stop-main_time - len(m_fit)*sum_time/(f+1)}")
    # final results presentation
    plt.figure()
    plt.plot(x_data, y_data, linewidth=0.5)
    plt.plot(x_data, lor, linewidth=0.5)
    plt.plot(x_data, [(abs(y_data[n]-lor[n]))
                      for n in range(w_len)], linewidth=0.5)
    plt.legend(["Data points", "Fit", "Absolute error"])
    print(f"main: Time consumption: {time.process_time()-start}")
    logging.info(f"main: Total time consumption: {time.process_time()-start}")
    plt.figure()
    # print(new_fit[-1])
    plt.plot(range(len(new_fit[-1])), [math.log(nf) for nf in new_fit[-1]])
    plt.legend(["Error level"])
    plt.xlabel("Step")
    plt.ylabel("log(Error)")
    if graphic:
        plt.show()
    return new_fit[0]


def set_globals(numb_peaks=1):
    """
    `Algo` special function: `set_globals`
    ---
    """
    # for key, value in kwargs.items():
    #     global
    global N_PEAKS
    N_PEAKS = numb_peaks
    return f"Number of peaks: {N_PEAKS}\n"


if __name__ == "__main__":
    from new_hope import w, i
    main(list(w), list(i), rate=0.8, scales=N_PEAKS*[0.5, 1, 2])
