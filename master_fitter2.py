# %%
import time as tm
import json as js
import numpy as np
import pandas as pd
import os.path as op
import argparse as ap
import spectrals as sp
import data_import as di
import matplotlib.pyplot as plt
from nano.functions import xpeak
from scipy.optimize import least_squares, leastsq

global BASE, SIGNAL, BASE_PARAMS, SGNL_PARAMS
global X_AV, SGNL_AV, SGNL_STD
global BANDS, THETA
global MATERIAL


def main(tag, *args, **kwargs):
    global BANDS, BASE_CTR, SGNL_CTR, MATERIAL
    material_conf = kwargs['materials'][tag]
    global_conf = kwargs['globals']
    BANDS = material_conf['bands']
    BASE_CTR = material_conf['curvesCount']['base']
    SGNL_CTR = material_conf['curvesCount']['signal']
    path = op.join(material_conf['path'], material_conf['direction'])
    if global_conf['mainPath'] != '':
        path = op.join(global_conf['mainPath'], path)
    di.PATH = path
    di.XNAME = global_conf['xName']
    di.YNAME = global_conf['yName']
    di.read_dir()
    di.get_data()
    di.FILTER = slice(*material_conf['filter'])
    MATERIAL = material_conf['name']


def compound_plot(x, pars, bands=None):
    rc_color = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plt.plot(x, sp.spectrum(x, params), color=next(rc_color))
    bands = np.array(bands, dtype=bool)
    # setting band selectors
    if bands is None:
        bands = np.array([np.linspace(0, 0, int(len(params)/3), dtype=bool)])
    # acquiring color for each existing band
    band_colors = [next(rc_color) for _ in bands]
    # + no band color
    band_colors = np.array([[0.3, 0.3, 0.3]]+band_colors)
    # reshaping bands selector with transpose
    for band, a, h, x0 in zip(bands.T, pars[::3], pars[1::3], pars[2::3]):
        counter_band = 1-sum(band)
        band = np.array([counter_band]+list(band), dtype=bool)
        plt.plot(x, sp.spectrum(x, [a, h, x0]), lw=0.7,
                 color=band_colors[band][0])


def selector(params, sd_interval=3):
    params = np.array(params)
    amps2 = params[0::3]**2
    hmfws2 = params[1::3]**2
    x0s2 = params[2::3]**2
    phase_vecs = np.log10(amps2 + hmfws2 + x0s2)
    phase_vecs_av = np.mean(phase_vecs)
    phase_vecs_sd = np.std(phase_vecs)
    upper = (phase_vecs > phase_vecs_av-sd_interval*phase_vecs_sd)
    lower = (phase_vecs < phase_vecs_av+sd_interval*phase_vecs_sd)
    ftr = (upper.astype(int)*lower.astype(int)).astype(bool)
    plt.plot(phase_vecs, marker='.')
    plt.plot([0, len(amps2)], [phase_vecs_av, phase_vecs_av], linestyle='--')
    plt.plot([0, len(amps2)], [phase_vecs_av+sd_interval*phase_vecs_sd,
                               phase_vecs_av+sd_interval*phase_vecs_sd],
             linestyle=':')
    plt.plot([0, len(amps2)], [phase_vecs_av-sd_interval*phase_vecs_sd,
                               phase_vecs_av-sd_interval*phase_vecs_sd],
             linestyle=':')
    plt.show()
    return np.array([ftr[m] for n in range(3) for m in range(len(ftr))])


# # %%
print("Data preparations... ")
_t0 = tm.time()
if __name__ == "__main__":
    # argument parsing
    args = None
    config_name = None
    tag = None
    try:
        parser = ap.ArgumentParser()
        parser.add_argument("-l", "--load",
                            help="Get settings from config JSON file",
                            type=str)
        parser.add_argument('-s', '--store',
                            help="Store acquired data from fitting",
                            action='store_true')
        parser.add_argument('-t', '--tag',
                            help="Tag of material configuration in "
                                 " config JSON", type=str)
        args = parser.parse_args()
        config_name = args.load
        tag = args.tag
    except SystemExit as e:
        print(e)
        while True:
            q = input("Would you like to specify config file name? Y/N")
            if q in ['y', 'Y']:
                config_name = input("Name: ")
                break
            elif q in ['n', 'N']:
                break
        if config_name is not None:
            tag = input("Config tag: ")

    # argument parsing aftermath
    if config_name is not None:
        try:
            confile = open(config_name, "r")
            config = js.load(confile)
            confile.close()
        except TypeError as e:
            print("Expected user input with `[--load|-l] *.json`, but got "
                  "none. Reverting to default behaviour instead.")
            __name__ = 'user_script'
        try:
            if tag is not None:
                # Calling main function, that sets up globals
                main(tag, **config)
            else:
                raise TypeError(f"Tag not selected. Value: {tag}")
        except TypeError as e:
            err_tag_selection = list(config['materials'].keys())
            raise NameError(f"Required tag from: {err_tag_selection}. "
                            f"Parsed tag: `{tag}`.")
    else:
        __name__ = "wrong_args"
if __name__ != '__main__':
    # global name
    MATERIAL = input("Specify material name: ")
    # data reading
    di.PATH = input("Data folder: ")
    _band_no_ = int(input("Number of bands (integer): "))
    di.XNAME = input("X column name: ")
    di.YNAME = input("Y column name: ")
    di.read_dir()
    di.get_data()
    # creating filter AFTER first data extractions to utilize di.X
    ranges = []
    print("X arg ranges:")
    ranges.append(float(input("\tleft: ")))
    ranges.append(float(input("\tright: ")))
    _ftr_ = (di.X > ranges[0]).astype(int)*(di.X < ranges[1]).astype(int)
    _ftr_ = _ftr_.astype(bool)
    ix_possible = np.arange(0, len(di.X), 1)
    di.FILTER = slice(ix_possible[_ftr_][0], ix_possible[_ftr_][-1])
    # filtering does not affect di.DATA, so all data is available at any moment
    # the only thing needed to regain access to filtered data from di.get_data
    # is changing filter

    BANDS = {
        'center': [None for _ in range(_band_no_)],
        'width': [None for _ in range(_band_no_)],
        'intensity': [0 for _ in range(_band_no_)]
    }

    # number of curves per part
    BASE_CTR = int(input("Number of signal curves (integer): "))
    SGNL_CTR = int(input("Number of baseline curves (integer): "))


# globals
# data globals
BASE = {}
SIGNAL = {}
# param globals
BASE_PARAMS = {}
SGNL_PARAMS = {}
# angles
global THETA
dth = np.pi/18
THETA = np.arange(0, 2*np.pi+dth, dth)

# base - signal separation
_ta = tm.time()
for nr, fname in enumerate(di.FILES):
    _tb = tm.time()
    eta = round((len(di.FILES)-nr-1)*(_tb-_ta)/(nr+1), 3)
    print(f"\tCurrent data: {nr+1}/{len(di.FILES)}\tETA: {eta}", end="\r")
    # base lines nad signals
    di.get_data(fname)
    BASE[fname] = sp.base_line(di.X, di.Y, av_scale=60)
    SIGNAL[fname] = di.Y - BASE[fname]

# averages
SGNL_AV = np.mean(list(SIGNAL.values()), axis=0)
SGNL_STD = np.std(list(SIGNAL.values()), axis=0)
X_AV = np.mean([di.DATA[fn][di.XNAME][di.FILTER] for fn in di.FILES],
               axis=0)

_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")

# # %%
# curves for base line
print("Curves for base line... ", end="")
_t0 = tm.time()
# initializing hmfws and locations
hmfw = (max(di.X)-min(di.X))/BASE_CTR
dx0 = (max(di.X)-min(di.X))/(BASE_CTR-1)
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

for _ in range(SGNL_CTR):
    # scaled parameters
    amp = 3*max(residuum)/4
    left, x0_ix, right = xpeak(X_AV, residuum, amp, amp/2)
    hmfw = abs(X_AV[left]-X_AV[right])
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
    print(f"\tCurrent data: {nr+1}/{len(di.FILES)}\tETA: {eta}", end="\r")
    _par, h = leastsq(sp.residual(*di.get_data(fl), prec=sp.f64),
                      list(BASE_PARAMS[fl])+list(params))
    SGNL_PARAMS[fl], BASE_PARAMS[fl] = _par[_base_p_ln:], _par[:_base_p_ln]
    BASE[fl] = sp.spectrum(di.X, BASE_PARAMS[fl])
    SIGNAL[fl] = di.Y - BASE[fl]
_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")
# # %%
print("Selection of bands by highest value... ")
_t0 = tm.time()
# selection algorithm, number of bands defined by initialization
sgnl_av_conv = sp.convolve(sp.kernel()(5), X_AV, SGNL_AV)
for nr, (center, width) in enumerate(zip(BANDS['center'], BANDS['width'])):
    left, center, right = xpeak(X_AV, sgnl_av_conv, max(sgnl_av_conv),
                                np.mean(sgnl_av_conv))
    if center is None:
        # highest band selection
        # setting bands parameters
        BANDS['center'][nr] = X_AV[center]
    if width is None:
        BANDS['width'][nr] = abs(X_AV[left]-X_AV[right])
    # suppressing already selected band
    sgnl_av_conv[left:right] = np.mean(sgnl_av_conv)

_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")
# # %%
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
        amps = np.abs(SGNL_PARAMS[fname][0::3])
        hmfws = np.abs(SGNL_PARAMS[fname][1::3])
        BANDS['intensity'][-1].append(amps[ftr].dot(hmfws[ftr]))
    BANDS['intensity'][-1] = np.array(BANDS['intensity'][-1])
_t1 = tm.time()
print(f"Done ({round(_t1-_t0, 3)}sec.)")

# %%
