# %%
from scipy.optimize import least_squares, leastsq
from nano.functions import d, div

from numpy import fft as nft

import matplotlib.pyplot as plt
import plotly.subplots as psp
import plotly.express as pex
import os.path as op

import pandas as pd
import numpy as np
import re
import os

plt.style.use('dark_background')

# Definitions
f128 = np.float128
f64 = np.float64
mlp = np.multiply
dvd = np.divide
pwr = np.power
add = np.add
sbt = np.subtract


def kernel(ktype='gauss', unitary=True, prec=f64):

    def _gauss_(sig):
        """Gaussian curve generator"""
        amp = unitary*((2*np.pi*pwr(sig, 2))**-0.5) + (not unitary)*1

        def _gauss_f_(x):
            """Gaussian: exp(- 1/2 * x**2 )"""
            return amp * np.exp(-0.5*(x/sig)**2)
        return _gauss_f_

    def _lorentz_(hmfw):
        """Lorentzian curve generator"""
        amp = unitary * dvd(1, (np.pi*hmfw)) + (not unitary)*1

        def _lorentz_f_(x):
            """Lorentzian: 1/( 1 + x**2 )"""
            return dvd(amp, (1 + (x/hmfw)**2))
        return _lorentz_f_

    def _fddens_(slope):
        """Logistic (Fermi-Dirac dist.) curve generator"""
        amp = unitary * slope + (not unitary)*4

        def _fddens_f_(x):
            """Logistic: 4/( exp(-x) + 2 + exp(x) )"""
            return dvd(amp, (np.exp(-slope*x) + 2 + np.exp(slope*x)))
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
    return np.array(sig_conv, dtype=f128)


def deconvolve(kernel, x, signal) -> np.array:
    # getting possibly most regular kernel (centered)
    # x = np.array(x, dtype=f128)
    x_center = (max(x) + min(x))/2
    k = kernel(x - x_center)
    # kernel fft
    kf = nft.fft(k)
    # data fft
    signal_f = nft.fft(signal)
    # equidistantly convolved kernel-anti-convolved ifft-ed data
    # ytac = IFT(FT(\int_{-\inf}^{\inf} y k_{s}(t-x) dx )/FT(k_{h}(t)))
    signal_ac = nft.fftshift(nft.ifft(signal_f/kf))
    # signal normalization constant
    signal_ac_sfc = sum(np.abs(signal_ac)*np.abs(d(x)))
    # normalization and renormalization
    return signal_ac * sum(signal*np.abs(d(x)))/signal_ac_sfc


def conv_variance(kernel, x, signal) -> np.array:
    yav2 = (convolve(kernel, x, signal, adj=True))**2
    y2av = (convolve(kernel, x, signal**2, adj=True))
    return y2av - yav2


def spectrum(x, param_vec, func=kernel('lorentz', unitary=False)) -> np.array:
    result = 0
    for n in range(0, len(param_vec)-2, 3):
        amp, shape_param, x0 = param_vec[n:n+3]
        result += abs(amp)*func(shape_param)(x-x0)
    return np.array(result)


def residual(x, y, weights=None, func=spectrum):
    if weights is None:
        weights = np.linspace(1, 1, len(x))

    def _res_(param_vec) -> np.array:
        return np.array((y-spectrum(x, param_vec))/weights)
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


def pex_fig(traces):
    fig = pex.scatter()
    fig.layout.template = 'plotly_dark'
    fig.add_traces(traces)
    layout = {'legend': {'traceorder': 'reversed'}}
    fig.update_layout(layout)
    fig.show()


def widen(x, num=0, ix=0):
    """Widening array by `num` elements, starting from `ix`"""
    x_f128 = np.array(x, dtype=f128)
    dx = np.mean(np.diff(x_f128))
    if num < 0:
        raise ValueError("Number of added elements `num` cannot be less"
                         " than 0")
    # if ix is non negative
    if 0 <= ix:
        x_stop = x_f128[-1]
        x_f128 = np.append(x_f128, np.linspace(x_stop+dx, x_stop+num*dx, num))
    # if less than zero, split linspaces
    elif num+ix >= 0:
        x_start = x_f128[0]
        x_stop = x_f128[-1]
        left_lspace = np.linspace(x_start+ix*dx, x_start-dx, abs(ix),
                                  dtype=f128)
        right_lspace = np.linspace(x_stop+dx, x_stop+dx*(num+ix), num+ix,
                                   dtype=f128)
        x_f128 = np.append(np.append(left_lspace, x_f128), right_lspace)
    else:
        raise ValueError("Number of added elements `num` cannot be smaller"
                         " than absolute value of `ix`")
    return np.array(x_f128, dtype=x.dtype)


# # %%
# Data files reading
path = ".data/Praca_inzynierska/Badania/200924/polar_si/VV"
data = {}
path, dir, files = next(os.walk(path))
for fname in files:
    data[fname] = pd.read_csv(op.join(path, fname), sep=r"\t{1,}",
                              header=0, engine='python')

# # %%
# Data extraction
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

# # %%
# Equidistancing data
# Convolution of data against equidistant x points can be
# performed with acceptably high accuracy because original
# x arg is semi-equidistant

# getting new x args (equidistant)
xe = np.array(np.linspace(min(x), max(x), len(x)))
# boundaries for extendes domain
xc = (max(x)+min(x))/2.0
dx = np.mean(np.diff(xe))
# new range (scaled)
scale = 2
t = widen(xe, len(xe), -int(len(xe)/2))

# convolving data to fit equidistant x arg
hmfw_equid = 1
kernel_equid_type = 'gauss'
ye = deconvolve(kernel(kernel_equid_type)(hmfw_equid), xe,
                convolve(kernel(kernel_equid_type)(hmfw_equid),
                         x, y, adj=True, t=xe))
ye = np.abs(ye)

# %%
