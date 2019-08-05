"""
This is main algorithm for finding fit of series of Lorentz's curves
to a given spectral data, especially Raman.

Following script had been written by rules of pylama code audit (W291)
Python 3.6.8 (64-bit) and with Visual Studio Code, therefore it gives no
errors nor warnings with this setup.
"""

import matplotlib.pyplot as plt
from __Main import wave             # wave data for specified exps
from __Main import inte             # intensity data
import DataStruct as ds             # to use some of functions
import math
import time                         # estimating elapsed time
import logging                      # for logging

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


def lorentz(A, B, x_arg, x0=0.0):
    """
    `Algo` method
    ---
    Description:
    ---
    Lorentz's function generator. Uses three paramters `A`, `B`, `x0` to
    return value of Lorentz's function at point specified in `x_arg`.
    """
    return A/((2*(x_arg-x0)/B)**2.0 + 1.0)


def array_it(function):
    """
    `array_it` function
    ---
    Description:
    ---
    Creating element-wise function object, which then might be used on
    iterator argument. Argument specified in `arg_name`:str

    Parameters:
    ---
    + `function` -- non-iterating function object

    Warning:
    ---
    Listing argument has to be named!
    """
    logging.debug(f"array_it({type(function)}: {function.__name__})")

    def release(P, Q):
        P.update(Q)
        return P

    def list_fun(*args, **kwargs):
        args = list(args)
        varnames = function.__code__.co_varnames
        for n in range(len(args)):
            if ('__len__' in dir(args[n])) and (type(args[n]) is not str):
                key_lt = varnames[n]
                x_list = args[n]
            else:
                kwargs = release(kwargs, {varnames[n]: args[n]})
        return [function(**(release(kwargs, {key_lt: x}))) for x in x_list]
    return list_fun


lorentz_arr = array_it(lorentz)  # creating arrayed function of lorentz


def peak_find(y_arg):
    """
    `Algo` method
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
    return (lambda b, x, x0: lorentz_arr(amp, b, x, x0),
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


def observe(y_arg, x_arg):
    """
    `Algo` method
    ---
    Description:
    ---
    Swipes through data, finds peaks by using `peak_find` method, and matches
    certain parameters of each Lorentz to finally return n-row 3-element list
    of parameters in the following order: [amplitude, half-amp width, peak arg]
    """
    logging.debug(f"observe({type(y_arg)}, {type(x_arg)})")
    sum_time = 0.0
    data_set = []
    for akn in range(N_PEAKS):                      # find N_PEAKS nr of curves
        start = time.process_time()
        peaks = peak_find(y_arg)                      # peak finding
        b_val = 0
        for m in range(1, 10000, 10):               # course-search for-loop
            s = peaks[0](m, x_arg, x_arg[peaks[1]])     # lorentz_arr
            ch_val = ds.pearson(y_arg, s)           # Pearson's R value check
            if ch_val > b_val:
                b_val = ch_val
            else:
                b_val = m
                break
        fine_val = 0
        for m in range(len(EY)):                    # fine-search for-loop
            s = peaks[0](b_val+EY[m], x_arg, x_arg[peaks[1]])
            ch_val = ds.pearson(y_arg, s)
            if ch_val > fine_val:
                fine_val = ch_val
            else:
                fine_val = b_val+EY[m]
                break
        data_set.append([peaks[2], fine_val, x_arg[peaks[1]]])  # main dataset
        lor = lorentz_arr(peaks[2], fine_val, x_arg, x_arg[peaks[1]])
        # y_arg change to eliminate found peaks
        y_arg = [y_arg[n]-lor[n] for n in range(len(y_arg))]
        logging.info(f"observe: step {akn}: {time.process_time()-start}\ts")
        sum_time += time.process_time()-start
    logging.info(f"Time consumption: {sum_time}.")
    return data_set


def main():
    """
    `Algo` main method
    ---
    Description:
    ---
    Uses `Algo.observe` method to fit Lorentz's curves to data, then
    generates fit points from return of `observe`. For all steps it
    calculates time needed to complete the task. Finally it presents
    results in form of a graph with three lines:
    """
    start = time.process_time()
    l1 = observe(inte[0], wave[0])                  # main algorithm call
    w_len = len(wave[0])
    lor = [0 for n in range(w_len)]                # init of estimation dataset

    # this gonna take some time, but also it will count it!
    sum_time = 0.0
    main_time = time.process_time()

    for l in range(len(l1)):             # for-loop to sum all Lorentz's
        step = time.process_time()
        for n in range(w_len):
            lor[n] += lorentz(l1[l][0],
                              l1[l][1],
                              wave[0][n],
                              l1[l][2])
        stop = time.process_time()
        sum_time += stop-step
        logging.info(f"Approximation ETA:"
                     f" {stop-main_time - len(l1)*sum_time/(l+1)}")

    # final results presentation
    plt.plot(wave[0], inte[0], linewidth=0.5)
    plt.plot(wave[0], lor, linewidth=0.5)
    plt.plot(wave[0], [inte[0][n]-lor[n] for n in range(w_len)], linewidth=0.5)
    plt.legend(["Data points", "Fit", "Error"])
    print(f"Time consumption: {time.process_time()-start}")
    logging.info(f"Time consumption: {time.process_time()-start}")
    plt.show()


def set_globals(numb_peaks=1):
    """
    `Algo` method
    """
    # for key, value in kwargs.items():
    #     global
    global N_PEAKS
    N_PEAKS = numb_peaks
    return f"Number of peaks: {N_PEAKS}\n"


# if __name__ == "__main__":
#     main()
