# %%
from scipy.optimize import least_squares, leastsq
from nano.functions import d, div
from numpy import fft as nft

import matplotlib.pyplot as plt
import os.path as op
import pandas as pd
import numpy as np
import re
import os

# while True:
#     try:
#         from scipy.optimize import least_squares
#         from scipy.optimize import leastsq
#         from nano.functions import d, div
#         from numpy import fft as nft

#         import matplotlib.pyplot as plt
#         import os.path as op
#         import pandas as pd
#         import numpy as np
#         import re
#         import os
#         break
#     except ModuleNotFoundError as e:
#         import subprocess as sp
#         module_name = str(e).split("No module named ")
#         if len(module_name) == 2:
#             module_name = module_name[1][1:-1]
#             bash_cmd = f"python3 -m pip install {module_name}"
#             proc = sp.Popen(bash_cmd.split(), stdout=sp.PIPE)
#             out, err = proc.communicate()
#             if err:
#                 raise Exception(err)
#         else:
#             raise e

plt.style.use('dark_background')


def kernel(ktype='gauss', unitary=True):

    def _gauss_(sig):
        """Gaussian curve generator"""
        amp = unitary*((2*np.pi*sig**2)**-0.5) + (not unitary)*1

        def _gauss_f_(x):
            """Gaussian: exp(- 1/2 * x**2 )"""
            return amp * np.exp(-0.5*(x/sig)**2)
        return _gauss_f_

    def _lorentz_(hmfw):
        """Lorentzian curve generator"""
        amp = unitary * 1/(np.pi*hmfw) + (not unitary)*1

        def _lorentz_f_(x):
            """Lorentzian: 1/( 1 + x**2 )"""
            return amp/(1 + (x/hmfw)**2)
        return _lorentz_f_

    def _fddens_(slope):
        """Logistic (Fermi-Dirac dist.) curve generator"""
        amp = unitary * slope + (not unitary)*4

        def _fddens_f_(x):
            """Logistic: 4/( exp(-x) + 2 + exp(x) )"""
            return amp/(np.exp(-slope*x) + 2 + np.exp(slope*x))
        return _fddens_f_

    class _kernel_:
        gauss = _gauss_
        lorentz = _lorentz_
        fddens = _fddens_

    return getattr(_kernel_, ktype)


def convolve(kernel, x, signal, adj=False, t=None, adjuster=None) -> np.array:
    sig_conv = []
    if t is None:
        t = x.copy()
    if adjuster is not None:
        for ti in t:
            kernel_dint = kernel(x-ti)*np.abs(d(x))
            kernal_int = sum(adjuster(x-ti)*np.abs(d(x)))
            sig_conv.append(sum(signal*kernel_dint/kernal_int))
        return np.array(sig_conv)
    if adj:
        for ti in t:
            kernel_dint = kernel(x-ti)*np.abs(d(x))
            kernal_int = sum(kernel_dint)
            sig_conv.append(sum(signal*kernel_dint/kernal_int))
    else:
        for ti in t:
            kernel_dint = kernel(x-ti)*np.abs(d(x))
            sig_conv.append(sum(signal*kernel_dint))
    return np.array(sig_conv)


def spectrum(x, param_vec, func=kernel('lorentz', unitary=False)):
    result = 0
    for n in range(0, len(param_vec)-2, 3):
        amp, shape_param, x0 = param_vec[n:n+3]
        result += amp*func(shape_param)(x-x0)
    return result


def residual(x, y, weights=None, func=spectrum):
    if weights is None:
        weights = np.linspace(1, 1, len(x))

    def _res_(param_vec):
        return (y-spectrum(x, param_vec))/weights
    return _res_


def cutoff(u, threshold, level=None):
    ix = np.arange(len(u))
    if level is None:
        level = threshold
    u_dict = dict(zip(ix, u))
    u_check = u > threshold
    up = [level for ui in u[u_check]]
    u_dict = {**u_dict, **dict(zip(ix[u_check], up))}
    return np.array(list(u_dict.values()))


def fft_ready(x: np.array, y: np.array):
    frq = 1/abs(np.mean(np.diff(x)))
    frng = int(len(y)/2)
    ft = nft.fft(y)/len(y)
    yft = ft[:frng]
    f = np.arange(0, frng, 1)*frq/frng
    return f, yft


def fft_plot(x: np.array, y: np.array, name='FFT', title='', scales="log-log"):
    f, yft = fft_ready(x, y)
    plt.plot(f, np.abs(yft), label=name, lw=0.7)
    plt.legend()
    scls_args = scales.split("-")
    scale_exc = "Wrong {axis} scale type: {value}"
    if not len(scls_args) < 2:
        if scls_args[0] == 'log':
            plt.xscale('log')
        elif scls_args[0]:
            raise ValueError(scale_exc.format(axis='x', value=scls_args[0]))
        if scls_args[1] == 'log':
            plt.yscale('log')
        elif scls_args[1]:
            raise ValueError(scale_exc.format(axis='x', value=scls_args[0]))
    elif scales:
        raise ValueError("Scales argument for x and y axes invalid, "
                         f"valid values: 'log-log', '-log', 'log-', "
                         f"'', got: {scales}")
    plt.xlabel('Freq')
    plt.ylabel('|Y(f)|')
    plt.title(title)


# %%
path = ".data/Praca_inzynierska/Badania/200924/polar_si/VV"
data = {}
path, dir, files = next(os.walk(path))
for fname in files:
    data[fname] = pd.read_csv(op.join(path, fname), sep=r"\t{1,}",
                              header=0, engine='python')

# %%
xnm = '#Wave'
ynm = '#Intensity'

mfile_no = 1

try:
    x = np.array(data[files[mfile_no]][xnm])
    y = np.array(data[files[mfile_no]][ynm])

    # plt.plot(x, y, '-', lw=0.7)
    # plt.xlabel("Wavenumber")
    # plt.ylabel("Intensity")
    # plt.show()

except IndexError:
    raise ValueError(f"Wrong number: mfile_no: {mfile_no}; min: 0; max: 36")

# %%

fig, ax = plt.subplots(nrows=2)
t = np.linspace(min(x), max(x), len(x))

yt = convolve(kernel()(0.1), x, y, adj=True, t=t)

ax[0].plot(x, y, '.', ms=1)
ax[0].plot(t, yt, '-', lw=0.7)

t_center = (max(t) + min(t))/2
k = kernel('lorentz')(1)((t - t_center))
kf = nft.fft(k)
ytf = nft.fft(yt)

ax[1].plot(t, d(nft.fftshift(nft.ifft(ytf/kf)))/d(t) - d(yt)/(t), '--', lw=0.5)
