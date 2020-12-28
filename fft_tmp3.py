# %%
from scipy.optimize import least_squares, leastsq
from nano.functions import d, div

from numpy import fft as nft

import matplotlib.pyplot as plt
import plotly.express as pex
import os.path as op

import pandas as pd
import numpy as np
import re
import os

plt.style.use('dark_background')

# Definitions


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
# Data files reading
path = ".data/Praca_inzynierska/Badania/200924/polar_si/VV"
data = {}
path, dir, files = next(os.walk(path))
for fname in files:
    data[fname] = pd.read_csv(op.join(path, fname), sep=r"\t{1,}",
                              header=0, engine='python')

# %%
# Data extraction
xnm = '#Wave'
ynm = '#Intensity'

mfile_no = 2

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
# Convolution of data against equidistant x points can be
# performed with acceptably high confidence because original
# x arg was semi-equidistant

traces_ytac = []
for hmfw in [1, 2, 5, 10, 20]:

    # getting new x args (equidistant)
    t = np.array(np.linspace(min(x), max(x), len(x)))
    # convolving data to fit equidistant x arg
    hmfw_equid = 1
    kernel_equid_type = 'gauss'
    yt = convolve(kernel(kernel_equid_type)(hmfw_equid), x, y, adj=True, t=t)
    yt = yt * sum(y*np.abs(d(x)))/sum(yt*d(t))

    # getting possibly most regular kernel (centered)
    t_center = (max(t) + min(t))/2
    kernel_type = 'lorentz'
    k = kernel(kernel_type)(hmfw)((t - t_center))*(np.pi*hmfw)
    # kernel fft
    kf = nft.fft(k)
    # data fft
    ytf = nft.fft(yt)
    # equidistantly convolved kernel-anti-convolved ifft-ed data
    # ytac = IFT(FT(\int_{-\inf}^{\inf} y k_{s}(t-x) dx )/FT(k_{h}(t)))
    ytac = np.abs(nft.fftshift(nft.ifft(ytf/kf)))
    # normalization and renormalization
    ytac = ytac * sum(y*d(t))/sum(ytac*d(t))
    # std dist of ytac with moving average
    hmfw_new = 0.01
    sytac_type = 'lorentz'
    sytac = (convolve(kernel(sytac_type)(hmfw_new), t, ytac**2, adj=True) -
             convolve(kernel(sytac_type)(hmfw_new), t, ytac, adj=True)**2)**0.5
    traces_ytac += [
        {
            'name': f"Anti-conv'd, hmfw: {hmfw}",
            'x': t,
            'y': ytac,
            'mode': 'lines',
            'line': {
                'width': 1
            }
        },
    ]

# plot
fig = pex.scatter()
fig.layout.template = 'plotly_dark'
traces = [
    {
        'name': 'Data',
        'x': x,
        'y': y,
        'mode': 'markers',
        'marker': {
            'size': 1
        }
    },
    {
        'name': 'Equidistant convolved',
        'x': t,
        'y': yt,
        'mode': 'lines',
        'line': {
            'dash': 'dash',
            'width': 1
        }
    },
    # {
    #     'name': 'Found peak',
    #     'x': t,
    #     'y': ytac,
    #     'mode': 'lines',
    #     'line': {
    #         'width': 1
    #     }
    # }
]

traces = traces_ytac + traces

layout = {
    'legend': {
        'traceorder': 'reversed'
    },
    # 'yaxis': {
    #     'range': [min(y)*0.9, max(y)*1.1]
    # }
}

fig.add_traces(traces[::-1])

fig.update_layout(layout)

fig.show()

# %%
