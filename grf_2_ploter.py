# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nano.data_import as di
import nano.spectrals as sp
from scipy.optimize import leastsq

style = 'default'
# style = 'dark_background'

plt.style.use(style)

img_path = "./.data/Praca_inzynierska/Badania/190522/grf/grf_2_2x50.bmp"

# Image
img_data = plt.imread(img_path)

# Raman
di.PATH = "./.data/Praca_inzynierska/Badania/190523/grf/grf514_2_2C.txt"
di.XNAME = "#Wave"
di.YNAME = "#Intensity"

di.read_dir(ext='txt')
di.get_data()

# %% Image
plt.imshow(img_data)

plt.ylim(800, 1200)
plt.xlim(700, 1100)

# vector
head = np.array([860, 1060])
pointer = [head-100, head]
plt.quiver(*pointer[0], *pointer[1],
           scale_units='xy', scale=10, angles='xy', color='black')
plt.xticks([])
plt.yticks([])
plt.show()

# %% Raman

s, a = 5, 50
print(s, a)
x, y = sp.equidist_data(di.X, di.Y, deconv=True)

plt.plot(x, y, '.', ms=1)
base = sp.base_line(x, y, sd_scale=s, av_scale=a)
plt.plot(x, base)
plt.show()
noise_lvl = y-sp.deconvolve(sp.kernel('lorentz')(1), x, y)
y_sd = sp.conv_variance(sp.kernel()(1), x, noise_lvl)**0.5

res_base = sp.residual(x, base, prec=np.float64)
p_init = np.array([max(base), 1000, np.mean(x)-500,
                   max(base), 1000, np.mean(x)+500,
                   max(base)/2, 200, np.mean(x)-400,
                   max(base)/2, 200, np.mean(x)+400]).astype(np.float64)
p, h = leastsq(res_base, p_init)

plt.plot(x, y)
plt.plot(x, sp.spectrum(x, p))
plt.show()
# %%
for amp, hmfw, x0 in zip(p[0::3], p[1::3], p[2::3]):
    plt.plot(x, sp.spectrum(x, [amp, hmfw, x0]), '--', lw=0.7)
plt.plot(x, y)
# %%
signal = y-sp.spectrum(x, p)
plt.plot(x, signal)

# %%
sfc_total = []
sfc_active = []

for amp, hmfw, x0 in zip(p[0::3], p[1::3], p[2::3]):
    amp = abs(amp)
    hmfw = abs(hmfw)
    sfc_total.append(np.pi*amp*hmfw)
    sfc_active.append(amp*hmfw*(np.arctan(max(di.X)/hmfw) -
                                np.arctan(min(di.X)/hmfw)))

plt.axis('equal')
plt.plot(sfc_total, sfc_active, '.')
for nr, st, sa in zip(range(len(sfc_total)), sfc_total, sfc_active):
    plt.text(st, sa, str(nr))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total')
plt.ylabel('Active')
plt.xlim(1e-3, 1e+10)
plt.ylim(1e-3, 1e+10)
# %%
