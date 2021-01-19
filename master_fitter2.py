# %%
import numpy as np
import matplotlib.pyplot as plt
import data_import as di
import spectrals as sp
import pandas as pd
from scipy.optimize import least_squares, leastsq
from nano.functions import xpeak
import time as tm


def compound_plot(x, params):
    plt.plot(x, sp.spectrum(x, params))
    for a, h, x0 in zip(params[::3], params[1::3], params[2::3]):
        plt.plot(x, sp.spectrum(x, [a, h, x0]), lw=0.7,
                 color=[0.3, 0.3, 0.3])


# # %%
print("Data preparations... ")
_t0 = tm.time()
# data reading
di.PATH = "./.data/Praca_inzynierska/Badania/190726/polar_grf/VH"
di.XNAME = '#Wave'
di.YNAME = '#Intensity'
di.read_dir()
di.get_data()

# globals
global BASE, SIGNAL
BASE = {}
SIGNAL = {}

global BASE_PARAMS, SGNL_PARAMS
BASE_PARAMS = {}
SGNL_PARAMS = {}

global BANDS, X_AV, SGNL_AV, SGNL_STD, THETA
BANDS = {
    'center': [-1, -1],
    'width': [-1, -1],
    'intensity': [0, 0]
}

# angles
dth = np.pi/18
THETA = np.arange(0, 2*np.pi+dth, dth)

# number of curves per part
base_curve_no = 4
sgnl_curve_no = 10
# base - signal separation
_ta = tm.time()
for nr, fname in enumerate(di.FILES):
    _tb = tm.time()
    eta = round((len(di.FILES)-nr-1)*(_tb-_ta)/(nr+1), 3)
    print(f"\tCurrent data: {nr+1}/{len(di.FILES)}\tETA: {eta}")
    # base lines nad signals
    di.get_data(fname)
    BASE[fname] = sp.base_line(di.X, di.Y, av_scale=60)
    SIGNAL[fname] = di.Y - BASE[fname]

# averages
SGNL_AV = np.mean(list(SIGNAL.values()), axis=0)
SGNL_STD = np.std(list(SIGNAL.values()), axis=0)
X_AV = np.mean([di.DATA[fn][di.XNAME] for fn in di.FILES], axis=0)

_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")

# %%
# curves for base line
print("Curves for base line... ", end="")
_t0 = tm.time()
# initializing hmfws and locations
hmfw = (max(di.X)-min(di.X))/base_curve_no
dx0 = (max(di.X)-min(di.X))/(base_curve_no-1)
x0s = np.arange(min(di.X), max(di.X)+dx0-0.1, dx0)

# reinventing bases in the terms of lorentzians
# and re-getting signals
for fl in di.FILES:
    # reading data, initializing params
    di.get_data(fl)
    amp = np.mean(BASE[fl])
    res = sp.residual(di.X, BASE[fl], prec=np.float64)
    p_init = np.zeros(3*len(x0s))
    p_init[0::3] = amp.copy()
    p_init[1::3] = hmfw.copy()
    p_init[2::3] = x0s.copy()
    # fitting params and getting new bases and signals
    p, h = leastsq(res, p_init)
    BASE[fl] = sp.spectrum(di.X, p)
    BASE_PARAMS[fl] = p.copy()
    SIGNAL[fl] = di.Y - BASE[fl]
_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")
# # %%
# finding parameters of average signal
print("Average model parameters... ")
_t0 = tm.time()
params = []
# initial residuum == signal
residuum = sp.residual(X_AV, SGNL_AV)([0, 1, 1])

for _ in range(sgnl_curve_no):
    # scaled parameters
    amp = 3*max(residuum)/4
    left, x0_ix, right = xpeak(X_AV, residuum, amp, amp/2)
    hmfw = 3*abs(X_AV[left]-X_AV[right])/4
    x0 = X_AV[x0_ix]
    # acquired parameters added
    params.append(amp.copy())
    params.append(hmfw.copy())
    params.append(x0.copy())
    # recalculating residuum
    residuum = sp.residual(X_AV, SGNL_AV)(params)

# updating params for higher accuracy
params, h = leastsq(sp.residual(X_AV, SGNL_AV, prec=np.float64),
                    np.array(params, dtype=np.float64))

_base_p_ln = len(BASE_PARAMS[di.FILES[0]])
_sgnl_p_ln = len(params)
_ta = tm.time()

# recalc of params for each dataset
# finalizing base-singal separation
for nr, fl in enumerate(di.FILES):
    _tb = tm.time()
    eta = round((len(di.FILES)-nr-1)*(_tb-_ta)/(nr+1), 3)
    print(f"\tCurrent data: {nr+1}/{len(di.FILES)}\tETA: {eta}")
    _par, h = leastsq(sp.residual(*di.get_data(fl), prec=sp.f64),
                      list(BASE_PARAMS[fl])+list(params))
    SGNL_PARAMS[fl], BASE_PARAMS[fl] = _par[_base_p_ln:], _par[:_base_p_ln]
    BASE[fl] = sp.spectrum(di.X, BASE_PARAMS[fl])
    SIGNAL[fl] = di.Y - BASE[fl]
_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")
# %%
print("Selection of bands by highest value... ")
_t0 = tm.time()
# selection algorithm, number of bands defined by initialization
sgnl_av_conv = sp.convolve(sp.kernel()(5), X_AV, SGNL_AV)
for nr, _ in enumerate(BANDS['center']):
    # highest band selection
    left, center, right = xpeak(X_AV, sgnl_av_conv, max(sgnl_av_conv),
                                np.mean(sgnl_av_conv))
    # setting bands parameters
    BANDS['center'][nr] = X_AV[center]
    BANDS['width'][nr] = abs(X_AV[left]-X_AV[right])
    # suppressing already selected band
    sgnl_av_conv[left:right] = np.mean(sgnl_av_conv)

_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")
# %%
print("Calculating band intensities... ")
_t0 = tm.time()
# creating band filters
for fname in SGNL_PARAMS.keys():
    for comp in SGNL_PARAMS.values():
        BANDS['filter'] = []
        for band, width in zip(BANDS['center'], BANDS['width']):
            right = np.array(comp[2::3] > band-width/2, dtype=int)
            left = np.array(comp[2::3] < band+width/2, dtype=int)
            BANDS['filter'].append(right*left)
# calculating intensities
BANDS['intensity'] = []
for ftr in BANDS['filter']:
    BANDS['intensity'].append([])
    for fname in SGNL_PARAMS.keys():
        amps = SGNL_PARAMS[fname][0::3]
        hmfws = SGNL_PARAMS[fname][1::3]
        BANDS['intensity'][-1].append(amps[ftr].dot(hmfws[ftr]))
    BANDS['intensity'][-1] = np.array(BANDS['intensity'][-1])
_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")

# %%
