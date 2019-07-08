import matplotlib.pyplot as plt
from __Main import wave
from __Main import inte
import DataStruct as ds
import math
import time

"""
This is main algorithm file. It takes data prepared in __Main
to further manage it
"""


def lorentz(A, B, x_arg, x0=0.0):
    result = ()
    for x in x_arg:
        result += (A/((2*(x-x0)/B)**2.0 + 1.0),)
    return result


def swipe(y_arg):
    data = y_arg
    data_c = y_arg.copy()
    data_c.reverse()
    minimum = ds.Helper.gravity_mean(data_c)
    amplitude = max(data)-minimum
    main_point = ds.nearest(data, max(data))
    # print(amplitude, main_point, minimum)
    lz = lorentz
    return (lambda b, x, x0: lz(amplitude, b, x, x0), main_point, amplitude)


X = [0.01*n for n in range(-1000, 1000)]
Y = [1-math.exp(-0.1*x**2) for x in X]
EY = [0.01*sum(Y[:n])-0.01*sum(Y)/2 for n in range(len(Y))]


def observe(y_arg, x_arg):
    sum_time = 0.0
    data_set = []
    for akn in range(20):
        start = time.process_time()
        swp = swipe(y_arg)
        b_val = 0
        for m in range(1, 10000, 10):
            s = swp[0](m, x_arg, x_arg[swp[1]])
            ch_val = ds.pearson(y_arg, s)
            if ch_val > b_val:
                b_val = ch_val
            else:
                b_val = m
                break
        fine_val = 0
        for m in range(len(EY)):
            s = swp[0](b_val+EY[m], x_arg, x_arg[swp[1]])
            ch_val = ds.pearson(y_arg, s)
            if ch_val > fine_val:
                fine_val = ch_val
            else:
                fine_val = b_val+EY[m]
                break
        data_set.append([swp[2], fine_val, x_arg[swp[1]]])
        lor = lorentz(swp[2], fine_val, x_arg, x_arg[swp[1]])
        y_arg = [y_arg[n]-lor[n] for n in range(len(y_arg))]
        print(time.process_time()-start)
        sum_time += time.process_time()-start
    print(f"Time consumption: {sum_time}")
    return data_set


start = time.process_time()
l1 = observe(inte[0], wave[0])
lor = [0 for n in range(len(wave[0]))]
sum_time = 0.0
main = time.process_time()
for l in range(len(l1)):
    step = time.process_time()
    lor = [lor[n] + lorentz(l1[l][0],
                            l1[l][1],
                            wave[0],
                            l1[l][2])[n] for n in range(len(wave[0]))]
    stop = time.process_time()
    sum_time += stop-step
    print(f"ETA: {stop-main - len(l1)*sum_time/(l+1)}")
plt.plot(wave[0], inte[0], linewidth=0.5)
plt.plot(wave[0], lor, linewidth=0.5)
print(f"Time consumption: {time.process_time()-start}")
plt.show()
