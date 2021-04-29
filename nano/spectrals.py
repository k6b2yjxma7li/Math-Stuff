# The Nano submodule

"""
`nano`'s `spectrals`
===
"""

from scipy.optimize import least_squares, leastsq
from nano.functions import d, div

from numpy import fft as nft

import matplotlib.pyplot as plt
import numpy as np


global MPL_STYLE
MPL_STYLE = 'dark_background'
plt.style.use(MPL_STYLE)

# Definitions
f128 = np.float128
f64 = np.float64
mlp = np.multiply
dvd = np.divide
pwr = np.power
add = np.add
sbt = np.subtract


def kernel(ktype='gauss', unitary=True, prec=f64):
    """kernel -- kernel function generator

    Parameters
    ----------
    ktype : str, optional
        kernel type; possible values: 'gauss', 'lorentz', 'fddens', by default 'gauss'
    unitary : bool, optional
        specifies if kernel is (surface) normalized or not, by default True
    prec : numpy numerical object, optional
        (data type) dtype of kernel results' numpy arrays, by default numpy.float64
    """
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
            return dvd(amp, (1 + (mlp(2, x)/hmfw)**2))
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


def convolve(kernel, x:np.ndarray, y:np.ndarray, t:np.ndarray=None,
              adj=True, adjuster=None, batch_size:int=None, prec=f64) -> np.ndarray:
    """Performance-optimised simple convolution of `kernel` over y-vector"""
    if t is None:
        t = x
    if adjuster is None:
        adjuster = kernel
    # 1st dim of matrices
    N = len(x)
    # size of t's elements (in bytes)
    B = t.itemsize
    # max 2nd size of matrices
    if batch_size is None:
        Mmax = int(2**(np.floor(np.log2((8*1024**2)/(N * B)))))
        # make sure Mmax != 0
        Mmax = Mmax + (Mmax < 1)
    else:
        Mmax = batch_size
    # yc init
    yc = np.zeros(len(t), dtype=prec)
    # x discrete differential
    dx = d(x)
    # if optim. conditions are fulfilled without chunking
    if t.size < Mmax:
        Mmax = t.size
    it = 0
    M = Mmax
    while yc.size > Mmax*it:
        if yc.size - Mmax*it > Mmax:
            left, right = Mmax*it, Mmax*(it+1)
        else:
            left, right = Mmax*it, yc.size
            M = yc.size - Mmax*it
        # construction of vectors' vector
        # shifting 0-point
        xmat_step = np.array([x,]*M) - np.array([t[left:right],]*N).T
        # kernel matrix
        kmat = kernel(xmat_step)
        if adj:
            # normalisation (to take into account boundary conditions)
            knorminv = np.array([1/(adjuster(xmat_step).dot(dx)),]*N).T
            # proper convolution
            yc[left:right] = (kmat * knorminv).dot(y*dx)
        else:
            yc[left:right] = kmat.dot(y*dx)
        it += 1
    return yc

def deconvolve(kernel, x: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Inverse of convolution"""
    # getting possibly most regular kernel (centered)
    # x = np.array(x, dtype=f128)
    x_center = (max(x) + min(x))/2
    if kernel is np.ndarray:
        k = kernel
    else:
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


def conv_variance(kernel, x, signal, adj=True, prec=f64) -> np.array:
    """Moving estimators' variance"""
    yav2 = (convolve(kernel, x, signal, adj=adj, prec=prec))**2
    y2av = (convolve(kernel, x, signal**2, adj=adj, prec=prec))
    return y2av - yav2


def spectrum(x, param_vec, func=None, prec=f128) -> np.array:
    """Spectrum -- sum of basic functions"""
    if func is None:
        func = kernel('lorentz', unitary=False, prec=prec)
    result = np.zeros(len(x), dtype=prec)
    for n in range(0, len(param_vec)-2, 3):
        amp, shape_param, x0 = param_vec[n:n+3]
        result += abs(amp)*func(shape_param)(x-x0)
    return np.array(result, dtype=prec)


def residual(x, y, weights=None, func=spectrum, prec=f128):
    """Compute weighted residuals of data vector `y` fitted by `func`"""
    x, y = x.astype(prec), y.astype(prec)
    if weights is None:
        weights = np.linspace(1, 1, len(x)).astype(prec)

    def _res_(param_vec, prec=prec) -> np.array:
        return np.array((y-spectrum(x, param_vec, prec=prec))/weights)
    return _res_


def cutoff(u, threshold, level=None):
    """Cutoff -- set values from `u` to some level if above `threshold`"""
    ix = np.arange(len(u))
    if level is None:
        level = threshold
    u_dict = dict(zip(ix, u))
    u_check = u > threshold
    up = [level for ui in u[u_check]]
    u_dict = {**u_dict, **dict(zip(ix[u_check], up))}
    return np.array(list(u_dict.values()))


def fft_ready(x: np.ndarray, y: np.ndarray):
    """Getting frequency domain and FFT of x and y data"""
    frq = 1/abs(np.mean(np.diff(x)))
    frng = int(len(y)/2)
    ft = nft.fft(y)/(len(y)/2)
    yft = ft[:frng]
    f = np.fft.fftfreq(len(x), 1/frq)[:frng]
    return f, yft


def fft_plot(x: np.array, y: np.array, name='FFT', title='', scales="log-log"):
    """Creating FFT plot from passed x and y data"""
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


def equidist_arg(t):
    """Generating equidistant points from arbitrary set"""
    return np.linspace(t[0], t[-1], len(t))


def equidist_data(u, v, kernel_eq=None, deconv=False):
    """Transformation of data points to fit equidistant positions"""
    t = equidist_arg(u)
    dt = np.mean(np.diff(t))
    if kernel_eq is None:
        # three sigma per one point
        kernel_eq = kernel()(dt/3)
    v_eq = convolve(kernel_eq, u, v, adj=True, t=t)
    if deconv:
        return t, deconvolve(kernel_eq, t, v_eq)
    else:
        return t, v_eq


def base_line(x, y, kern=kernel(), sd_scale=10, av_scale=100, adj=True):
    dx = np.mean(np.diff(x))
    y_sd = conv_variance(kern(sd_scale*dx), x, y, adj=adj)
    return convolve(lambda x: kern(av_scale*dx)(x)/y_sd, x, y, adj=adj)
