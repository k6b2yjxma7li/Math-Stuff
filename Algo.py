import matplotlib.pyplot as plt
from __Main import wave             # wave data for specified exps
from __Main import inte             # intensity data
import DataStruct as ds             # to use some of functions
import math
import time                         # estimating elapsed time

"""
This is main algorithm file. It takes data prepared in __Main
to further manage it as spectral data.

Following script had been written by rules of pylama code audit (W291)
Python 3.6.8 (64-bit) and with Visual Studio Code, therefore it gives no
errors nor warnings with this setup.
"""

global N_peaks
N_peaks = 20


def lorentz(A, B, x_arg, x0=0.0):
    """
    `Algo` method
    ---
    Description:
    ---
    Lorentz's function generator. Uses three paramters `A`, `B`, `x0` to
    return `tuple` of values from specified as `x_arg` set of arguments.
    """
    result = ()
    for x in x_arg:
        result += (A/((2*(x-x0)/B)**2.0 + 1.0),)
    return result


def peak_find(y_arg):
    """
    `Algo` method
    ---
    Description:
    ---
    Finds peak's parameters basing on `y_arg` argument containing spectral
    data. Returns three-element tuple:
    + 0: function wrapper for Lorentz with args: `b`, `x`, `x0`
    + 1: index of peak maximum in `y_arg`
    + 2: amplitude of peak
    """
    data = y_arg
    data_c = y_arg.copy()
    data_c.reverse()
    minimum = ds.Helper.gravity_mean(data_c)    # min to use as background
    amplitude = max(data)-minimum               # amplitude
    main_point = ds.nearest(data, max(data))    # finding peak point arg
    lz = lorentz                                # redefinition to shorten
    return (lambda b, x, x0: lz(amplitude, b, x, x0), main_point, amplitude)


"""
In the following section there are definitions for fine tuning of half-amp
peak width, aka `b` value.
"""

X = [0.01*n for n in range(-1000, 1000)]        # data range [-10;10]
Y = [1-math.exp(-0.1*x**2) for x in X]          # flipped Gauss curve from X
EY = [0.01*sum(Y[:n])-0.01*sum(Y)/2 for n in range(len(Y))]  # set to check

"""
Following method is main function, that calculates parameters of Lorentz's
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
    sum_time = 0.0
    data_set = []
    for akn in range(N_peaks):                      # finding 20 Lorentz curves
        start = time.process_time()
        swp = peak_find(y_arg)                      # peak finding
        b_val = 0
        for m in range(1, 10000, 10):               # course-search for-loop
            s = swp[0](m, x_arg, x_arg[swp[1]])     # lorentz thrown here
            ch_val = ds.pearson(y_arg, s)           # Pearson's r value check
            if ch_val > b_val:
                b_val = ch_val
            else:
                b_val = m
                break
        fine_val = 0
        for m in range(len(EY)):                    # fine-search for-loop
            s = swp[0](b_val+EY[m], x_arg, x_arg[swp[1]])
            ch_val = ds.pearson(y_arg, s)
            if ch_val > fine_val:
                fine_val = ch_val
            else:
                fine_val = b_val+EY[m]
                break
        data_set.append([swp[2], fine_val, x_arg[swp[1]]])  # main dataset
        lor = lorentz(swp[2], fine_val, x_arg, x_arg[swp[1]])
        # argument switch to eliminate found peaks
        y_arg = [y_arg[n]-lor[n] for n in range(len(y_arg))]
        print(time.process_time()-start)
        sum_time += time.process_time()-start
    print(f"Time consumption: {sum_time}")
    return data_set


def main():
    start = time.process_time()
    l1 = observe(inte[0], wave[0])              # main algorith call
    lor = [0 for n in range(len(wave[0]))]      # init of estimation dataset

    # this gonna take some time, but also it will count it!
    sum_time = 0.0
    main_time = time.process_time()
    for l in range(len(l1)):                    # for-loop to sum all Lorentzs
        step = time.process_time()
        lor = [lor[n] + lorentz(l1[l][0],
                                l1[l][1],
                                wave[0],
                                l1[l][2])[n] for n in range(len(wave[0]))]
        stop = time.process_time()
        sum_time += stop-step
        print(f"ETA: {stop-main_time - len(l1)*sum_time/(l+1)}")

    # final results presentation
    plt.plot(wave[0], inte[0], linewidth=0.5)
    plt.plot(wave[0], lor, linewidth=0.5)
    print(f"Time consumption: {time.process_time()-start}")
    plt.show()


def set_globals(numb_peaks=1):
    global N_peaks
    N_peaks = numb_peaks
