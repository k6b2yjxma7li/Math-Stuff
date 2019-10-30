from Algo import observe
from Algo import lorentz, lorentz2, lorentz3, error_gen
from Algo import N_PEAKS
import DataStruct as ds
from Marching_sphere import marching_sphere, sphere_generator
# from __Main import wave, inte
import matplotlib.pyplot as plt
import numpy as np
import glob
import math as m
import time
from __TEST_FUNS__ import * 

N_PEAKS = 10

path = "/home/rumcajs/Desktop/Praca_inzynierska/Badania/190725/polar_hbn/VV/"

polar_hbn = ds.DataStruct(path)
# # [ds.csv_convert(path=path, file_name=fname.split("/")[-1], new_path=path)
# #  for fname in glob.glob(path+"*.txt")]

polar_hbn.read_routine()

wavenumber = polar_hbn.data[0]["#Wave"]
intensity = polar_hbn.data[0]["#Intensity"]

vectors = observe(intensity, wavenumber, N_PEAKS)

l2 = []
[l2.extend([v[0], (2/v[1])**2, v[2]]) for v in vectors]

print("Initialization done.")
print("Marching the spheres...")

w = np.array(wavenumber)
i = np.array(len(w)*[0.0])
for lv in vectorize(l2, 3):
    i += lorentz2(*lv, w)
plt.figure()
plt.plot(w, i, linewidth=0.7)
plt.plot(wavenumber, intensity, '.', markersize=0.7)

new_lorentz = lambda a, b, x0, x: lorentz2(a, b, x0, x)

scales = N_PEAKS*[2, 5, 5]
steps = [scales[n]*l2[n] for n in range(len(scales))]

err_fun = error_gen(some_fun=new_lorentz,
                    y_data=intensity,
                    x_data=wavenumber,
                    start=l2)

d_err_fun = gradient(err_fun, 1e-6)

grad = d_err_fun(l2)

amp = [l2[n] for n in range(0, len(l2), 3)]
hw = [l2[n] for n in range(1, len(l2), 3)]
x0 = [l2[n] for n in range(2, len(l2), 3)]

plt.figure()
plt.plot(range(len(amp)), amp, '.')
plt.plot(range(len(hw)), hw, '+')
plt.plot(range(len(x0)), x0, 'o')

plt.show()

# t0 = time.process_time()

# grd_err = gradient(err_fun, 1e-9)

# t1 = time.process_time()
# a = np.array(grd_err(l2))

# t2 = time.process_time()
# b = np.array(len(l2)*[err_fun(l2)]) - np.array(l2)*a

# te = time.process_time()
# print((t0-t1)*1e+3,
#       (t1-t2)*1e+3,
#       (t2-te)*1e+3,
#       "ms")
