"""
_  _  _  __  ___ __  __
|\\/| /_\\ (_   |  |_  |_)
|  | | | __)  |  |_  | \\
__  _ ___ ___ ___ __
|_  |  |   |  |_  |_)
|   |  |   |  |__ | \\

Obviously too long script for fitting Raman spectrals and
extracting particular bands for polar analysis.
"""
# %%
# Imports and prerequisites
import os
import re
import sys
import warnings

import pandas as pd

import plotly.express as pex
import plotly.subplots as psp
import plotly.graph_objects as pgo

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from nano.table import table
from nano.addfun import gdev, lorentz, residual, spectrum
from nano.functions import d, nearest_val, pearson, smoothing, xpeak

if "--W" not in sys.argv:
    warnings.filterwarnings("ignore")
if "--light" not in sys.argv:
    plt.style.use('dark_background')
    plotly_global = 'plotly_dark'
else:
    plotly_global = 'plotly'
gav = smoothing

print(__doc__)

if "--light" in sys.argv:
    def glob_style(clr): return clr
else:
    def glob_style(clr): return np.array([1, 1, 1, 1])[:len(clr)]-np.array(clr)


def percentage(u, v):
    val = 100*u/v
    dig = int(abs(np.log10(val)-3))
    return f"{round(val, dig)}%"


def simple_compnts(sol, comp_nr=0):
    if comp_nr not in [0, 1, 2]:
        raise ValueError("Invalid argument comp_nr, proper values include 0, 1"
                         " or 2")
    return sol[comp_nr::3]


def compnts(sol_set, comp_nr=0):
    return [simple_compnts(s, comp_nr) for s in sol_set]


def gauss(t):
    return 1/(2*np.pi)**0.5 * np.exp(-0.5*(t)**2)


def av_param(value, param, xn, k):
    param = np.array(param)
    xn = np.array(xn)
    gaussian = 1/k * gauss((xn - value)/k)
    gaussian /= sum(gaussian)
    return sum(gaussian*param)


def mdv(v, u):
    v_cp = v.copy()
    while True:
        v_cp = d(v_cp)/d(u)
        yield v_cp


def transform(kernel, u, v):
    u = np.array(u)
    v = np.array(v)
    return (sum(v*kernel(u-ui)) for ui in u)


def kernel_gen(kernel, params):
    return lambda t: kernel(params, t)/sum(kernel(params, t))


def window(cutoff, center, width):
    return lambda slt: (np.exp(cutoff*np.array(slt) - center - width/2) +
                        np.exp(-cutoff*np.array(slt) + center + width/2))


def selector(vectors, sigma_no=3):
    vectors = np.array(vectors)
    vecs_norms = np.diag(vectors.T.dot(vectors))
    vecs_norm_av = np.mean(vecs_norms)
    vecs_norm_std = np.std(vecs_norms)
    selector_filter = np.array(vecs_norms < (vecs_norm_av +
                                             sigma_no*vecs_norm_std),
                               dtype=int)
    selector_filter *= np.array(vecs_norms > (vecs_norm_av -
                                              sigma_no*vecs_norm_std),
                                dtype=int)
    return selector_filter.astype(bool)


# %%
# Configuration
config = {
    'hbn': {
        'fullname': 'hBN',
        'init': 3,
        'count': 7,
        'day': '190725',
        'measure': 'polar_hbn',
        'frame': slice(0, -200),
        'bands': [1367],
        'M': {
            'init': 500,
            'final': 10
        },
        'solutions': {
            'VH': [],
            'VV': []
        },
        'average': {
            'VH': [],
            'VV': []
        },
        "polar": []
    },
    'grf': {
        'fullname': 'Graphene',
        'init': 5,
        'count': 12,
        'day': '190726',
        'measure': 'polar_grf',
        'bands': [1570, 1570, 2680, 2680],
        'frame': slice(0, -1),
        'M': {
            'init': 500,
            'final': 20
        },
        'solutions': {
            'VH': [],
            'VV': []
        },
        'average': {
            'VH': [],
            'VV': []
        },
        "polar": []
    },
    'si': {
        'fullname': 'Silicon',
        'init': 5,
        'count': 15,
        'day': '200924',
        'bands': [520, 520, 520],
        'frame': slice(0, -1),
        'measure': 'polar_si',
        'M': {
            'init': 5,
            'final': 5
        },
        'solutions': {
            'VH': [],
            'VV': []
        },
        'average': {
            'VH': [],
            'VV': []
        },
        'polar': []
    },
    'figsave': False
}

# %%
# Data loading and preparations
print("Data loading...", end="")
material = 'grf'
if '--direction' in sys.argv:
    dix = sys.argv.index('--direction')
    directions = [sys.argv[dix+1]]
elif '-d' in sys.argv:
    dix = sys.argv.index('-d')
    directions = [sys.argv[dix+1]]
else:
    directions = ['VV', 'VH']

df_rad = pd.DataFrame()

# main loop
for direct in directions:
    measure = config[material]['measure']
    day = config[material]['day']

    path = f"./.data/Praca_inzynierska/Badania/{day}/{measure}/{direct}"
    tdict = {}  # translation dict for accessing tables columns by integer
    tbl = table()
    _, _, files = next(os.walk(path))
    datafiles = []
    for n in range(len(files)):
        if ".txt" in files[n]:
            datafiles += [files[n]]
            dataset = re.findall(r"\d\d\d(?=[A-Z]{2})", files[n])
            tdict.update({n: dataset[0]})
    for nr, name in enumerate(datafiles):
        name = os.path.join(path, name)
        tbl += {tdict[nr]: table().read_csv(open(name, 'r'),
                delim=r",|\t{1,}")}
    datafiles = dict(zip(tdict.keys(), datafiles))
    tbl = tbl.sort()
    y_av = 0
    y_stdev = 0
    ctr = 0
    x_av = 0
    for key in tbl.keys():
        ctr += 1
        y_av += np.array(tbl[key]['#1'])
        y_stdev += np.array(tbl[key]['#1'])**2
        x_av += np.array(tbl[key]['#0'])
    y_av /= ctr
    y_stdev /= ctr
    y_stdev = (y_stdev - y_av**2)**0.5
    x_av /= ctr

    print("\tDone:")
    print(f"Material: {config[material]['fullname']}")
    print(f"Direction: {direct}")
    print(f"From 20{day[:2]}-{day[2:4]}-{day[4:6]}")
    if '--pitstop' in sys.argv:
        input()
    # %%
    # Main resgd fitter

    raman = spectrum(lorentz, 3)
    dif = residual(raman, x_av, y_av)
    ressq = residual(raman, x_av, y_av, 1/y_av**0.5)
    res = residual(raman, x_av, y_av, y_stdev)
    sol = []
    # Fitting with stdev
    u, v = x_av[config[material]['frame']], y_stdev[config[material]['frame']]

    dif = residual(raman, u, v)
    r = dif([1, 1, 0])

    sol = []

    print("Average model fitting... ", end="")

    y = y_av[config[material]['frame']]
    x = x_av[config[material]['frame']]
    initial = True
    dif = residual(raman, x, y)
    r = dif([0, 1, 1])

    def resg_gen(fn, u, v, w):
        def _res_(slt):
            return residual(fn, u, v, w)(slt)*np.array(slt).dot(np.array(slt))
        return _res_

    M = config[material]['M']['init']
    while len(sol)/3 < config[material]['init']:
        gd = (gdev(r, M)**2 + gav(r, M)**2)
        # haha resgd = residual(raman, x, y, 1/gd)
        p = xpeak(x, gd, max(gd), max(gd)/2)
        fwhm = abs(x[p[0]] - x[p[-1]])/2
        Amp = r[p[1]]
        x0 = (x[p[0]] + x[p[-1]])/2
        v = [abs(Amp), fwhm, x0]
        sol = list(sol)
        sol += [Amp, fwhm, x0]
        sol, h = leastsq(dif, sol)
        r = dif(sol)
        print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

    M = config[material]['M']['final']
    while len(sol)/3 < config[material]['count']:
        gd = (gdev(r, M)**2 + gav(r, M)**2)
        p = xpeak(x, gd, max(gd), max(gd)/2)
        fwhm = abs(x[p[0]] - x[p[-1]])/2
        Amp = r[p[1]]
        x0 = (x[p[0]] + x[p[-1]])/2
        v = [abs(Amp), fwhm, x0]
        sol = list(sol)
        sol += [Amp, fwhm, x0]
        sol, h = leastsq(dif, sol)
        r = dif(sol)
        print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")
    print("Final: ", end="")
    sol, h = leastsq(dif, sol)
    r = dif(sol)
    print(f"{percentage(r.dot(r), y.dot(y))}")
    print("Fitting: Done")
    if '--pitstop' in sys.argv:
        input()
    # %%
    # Active vs overall surface
    ix = 0
    while True:
        # new requirement
        requirement = selector([sol[0::3], sol[1::3], sol[2::3]], 3.3)
        new_req = []
        for n in range(len(requirement)):
            for k in range(3):
                new_req += [requirement[n]]
            if not requirement[n]:
                print(f"Trimmed: index {n}")
        new_req = np.array(new_req)
        sol = sol[new_req]
        if False not in requirement:
            print("No trimming")
            break
        ix += 1
    config[material]['average'][direct] = sol.copy()
    if '--show' in sys.argv:
        plt.show()
    if '--pitstop' in sys.argv:
        input()
    # %%
    # Main fitter plotter
    print("Plotting...")
    K = M
    gd = (gdev(r, M)**2 + gav(r, M)**2)
    r = dif(sol)

    fig = psp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                            specs=[[{'secondary_y': True}],
                                   [{'secondary_y': False}]])
    fig.layout.template = plotly_global

    traces1 = [
        {
            'type': 'scatter',
            'mode': 'markers',
            'name': 'data average',
            'marker': {
                'size': 2,
                'color': 'white'
            },
            'x': x,
            'y': y
        },
        {
            'type': 'scatter',
            'mode': 'lines',
            'name': 'data average',
            'line': {
                'width': 1,
                'color': 'cyan'
            },
            'x': x,
            'y': raman(sol, x)
        },
        {
            'type': 'scatter',
            'mode': 'markers',
            'name': 'data stdev',
            'marker': {
                'size': 1,
                'color': 'yellow'
            },
            'x': x,
            'y': y_stdev
        },
        {
            'type': 'scatter',
            'mode': 'lines',
            'name': 'penalty',
            'line': {
                'width': 0.1,
                'color': 'white'
            },
            'yaxis': 'y2',
            'x': x,
            'y': 1/gd
        }
    ]

    traces2 = [
        {
            'type': 'scatter',
            'mode': 'markers',
            'name': 'residuals',
            'marker': {
                'size': 1,
                'color': 'white'
            },
            'x': x,
            'y': r
        },
        {
            'type': 'scatter',
            'mode': 'lines',
            'name': 'residual average (full)',
            'line': {
                'width': 1,
                'color': 'white'
            },
            'x': x,
            'y': np.linspace(np.mean(r), np.mean(r), len(x))
        },
        {
            'type': 'scatter',
            'mode': 'lines',
            'name': 'residual average (full)',
            'line': {
                'width': 1,
                'color': 'yellow'
            },
            'x': x,
            'y': np.linspace(np.std(r), np.std(r), len(x))
        },
        {
            'type': 'scatter',
            'mode': 'lines',
            'name': 'residual average',
            'line': {
                'width': 0.2,
                'color': 'white'
            },
            'x': x,
            'y': smoothing(r, K)
        },
        {
            'type': 'scatter',
            'mode': 'lines',
            'name': 'residual stdev',
            'line': {
                'width': 0.2,
                'color': 'yellow'
            },
            'x': x,
            'y': gdev(r, K)
        },
    ]

    layout = {
        'width': 750,
        'height': 900,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1
        },
        'xaxis_title': 'wavenumber [cm^-1]',
        'yaxis_title': 'counts (data + fit + stdev)',
        'scene1': {
            'xaxis': {
                'title': 'wavenumber'
            },
            'yaxis': {
                'title': 'counts (data+fit+stdev)'
            }
        },
        'scene2': {
            'xaxis': {
                'title': 'wavenumber'
            },
            'yaxis': {
                'title': 'counts'
            }
        }
    }

    pltconf = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    }

    for trace in traces1:
        if 'yaxis' in trace:
            fig.add_trace(trace, row=1, col=1, secondary_y=True)
        else:
            fig.add_trace(trace, row=1, col=1, secondary_y=False)

    for trace in traces2:
        fig.add_trace(trace, row=2, col=1)
    fig.update_layout(layout)
    if '--show' in sys.argv:
        fig.show(config=pltconf)

    fig.write_html(f"./model_av_{material}_{direct}.html")

    fig = psp.make_subplots()

    fig.layout.template = plotly_global

    traces = [
        {
            'name': 'data',
            'type': 'scatter',
            'mode': 'markers',
            'marker': {
                'size': 2,
                'color': 'white'
            },
            'x': x,
            'y': y_av
        },
        {
            'name': 'fit',
            'type': 'scatter',
            'mode': 'lines',
            'line': {
                'width': 1,
                'color': 'cyan'
            },
            'x': x,
            'y': raman(sol, x)
        }
    ]

    pltconf = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    }

    spectrals = []

    for nr, n in enumerate(range(0, len(sol)-1, 3)):
        spectrals.append({
            'name': f'Comp. #{nr}',
            'text': f'A:{sol[n]}\nw:{sol[n+1]}\nx:{sol[n+2]}',
            'type': 'scatter',
            'mode': 'lines',
            'line': {
                'width': 0.5,
                'color': 'white'
            },
            'x': x,
            'y': lorentz(sol[n:n+3], x)
        })

    traces += spectrals

    layout = {
        'title': 'Spectral components for average model',
        'xaxis_title': 'wavenumber [cm^-1]',
        'yaxis_title': 'counts (data + fit + compnts)',
        'scene': {
            'xaxis': {
                'title': 'wavenumber'
            },
            'yaxis': {
                'title': 'counts (data+fit+stdev)'
            }
        },
        'margin': {
            't': 25,
            'b': 30,
            'l': 30,
            'r': 30
        },
    }

    for trace in traces:
        fig.add_trace(trace, row=1, col=1)

    fig.update_layout(layout)
    if '--show' in sys.argv:
        fig.show(config=pltconf)

    fig.write_html(f"./model_err_{material}_{direct}.html")
    if '--pitstop' in sys.argv:
        input()
    # %%
    # Proper logic (fitting all data sets)
    print("Directional data fitting...")
    for dset in tbl.keys():
        # Attempt to apply model to all data sets
        # dset = '#0' its the same '000'
        sol = config[material]['average'][direct]
        y = np.array(tbl[dset]['#Intensity'])[config[material]['frame']]
        sol, h = leastsq(residual(raman, x, y),
                         sol)
        dif = residual(raman, x, y)
        print(f"For {dset} deg.: {dif(sol).dot(dif(sol))}")

        ix = 0
        # Main resgd fitter
        M = config[material]['M']['init']
        while len(sol)/3 < config[material]['init']:
            resgd = residual(raman, x, y, 1/gd)
            p = xpeak(x, gd, max(gd), max(gd)/2)
            fwhm = abs(x[p[0]] - x[p[-1]])/2
            Amp = r[p[1]]
            x0 = (x[p[0]]+x[p[-1]])/2
            v = [abs(Amp), fwhm, x0]
            sol = list(sol)
            sol += [Amp, fwhm, x0]
            sol, h = leastsq(dif, sol)
            r = dif(sol)
            print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

        M = config[material]['M']['final']
        while len(sol)/3 < config[material]['count']:
            resgd = residual(raman, x, y, 1/gd)
            p = xpeak(x, gd, max(gd), max(gd)/2)
            fwhm = abs(x[p[0]] - x[p[-1]])/2
            Amp = r[p[1]]
            x0 = (x[p[0]]+x[p[-1]])/2
            v = [abs(Amp), fwhm, x0]
            sol = list(sol)
            sol += [Amp, fwhm, x0]
            sol, h = leastsq(dif, sol)
            r = dif(sol)
            print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")
            # new requirements, sig_no to be refined
            requirement = selector([sol[0::3], sol[1::3], sol[2::3]], 3.3)
            new_req = []
            for n in range(len(requirement)):
                for k in range(3):
                    new_req += [requirement[n]]
                if not requirement[n]:
                    print(f"Trimmed: index {n}")
            new_req = np.array(new_req)
            sol = sol[new_req]
            if False not in requirement:
                print("No trimming")
                break
            ix += 1
        config[material]['solutions'][direct].append(sol)
    if '--pitstop' in sys.argv:
        input()
    # %%
    # All data comparison
    print("Plotting...")
    sols = config[material]['solutions'][direct]

    # Create figure
    fig = pgo.Figure()
    fig.layout.template = plotly_global

    trace_count = []

    bands = config[material]['bands']
    ixs_all = []
    for ns, sol in enumerate(sols[:]):
        y = np.array(tbl[f'#{ns}']['#Intensity'])

        # Basic traces (always visible)
        traces = [
            {
                'name': 'data',
                'type': 'scatter',
                'mode': 'markers',
                'marker': {
                    'size': 2,
                    'color': 'white'
                },
                'visible': False,
                'x': x,
                'y': y
            },
            {
                'name': 'fit',
                'type': 'scatter',
                'mode': 'lines',
                'line': {
                    'width': 1,
                    'color': 'cyan'
                },
                'visible': False,
                'x': x,
                'y': raman(sol, x)
            }
        ]
        # bands selection mechanism
        ixs = []
        sol_cp = sol.copy()
        sol_ix = np.arange(int(len(sol)/3))  # keeping track of real ix
        # instead of dropped-element array ix-es
        threshold = 0.9
        for band in bands:
            ix = next(nearest_val(simple_compnts(sol_cp, 2), band))
            # making sure no over-the-top bands are selected
            # (selected - band) < 2*width
            w = simple_compnts(sol_cp, 1)[ix]
            x0 = simple_compnts(sol_cp, 2)[ix]
            # this if defines out of band rule
            if (abs(band - x0) > 5*abs(w) and
               5*abs(w) > abs(np.mean(np.diff(x_av)))):
                # selection rules for bands might be changed in next iterations
                # this one is not final
                continue
            # selecting preceding ix-s removed and using this as ix shift
            # dropping elements already selected (no double selection)
            ix_real = sol_ix[ix]
            sol_ix = np.delete(sol_ix, ix)
            for n in range(3):
                sol_cp = np.delete(sol_cp, 3*ix)
            ixs.append(ix_real)    # real ix-s
        ixs_all.append(ixs)

        for nr, s in enumerate([sol[n:n+3] for n in range(0, len(sol)-1, 3)]):
            comp_trace = {
                'visible': False,
                'text': f'A:{s[0]}\nw:{s[1]}\nx:{s[2]}',
                'line': {
                    'width': 0.1,
                    'color': 'white'
                },
                'name': f'Comp. #{nr}',
                'x': x,
                'y': lorentz(s, x)
            }
            if nr in ixs:
                comp_trace['line']['width'] = 1
            traces += [comp_trace]
        for trace in traces:
            fig.add_trace(trace)
        trace_count.append(len(traces))
    # Make first traces visible
    for n in range(trace_count[0]):
        fig.data[n].visible = True
    # Create and add slider
    steps = []
    for i in range(0, len(trace_count), 1):
        step = {
            'method': 'update',
            'args': [
                {'visible': [False]*len(fig.data)},
                {'title': f'{config[material]["fullname"]} {direct}'
                          f' @ Angle {i*10} deg.'}
            ]
        }
        for m in range(sum(trace_count[:i]), sum(trace_count[:i+1]), 1):
            step["args"][0]["visible"][m] = True
        steps.append(step)

    sliders = [{
        'active': 0,
        'currentvalue': {},
        'pad': {'t': 10},
        'steps': steps
    }]

    layout = {
        'title': f'{config[material]["fullname"]} {direct}'
    }

    pltconf = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    }

    fig.update_layout(layout, sliders=sliders)
    fig.update_yaxes(range=[0, 1.1*max(y_av+3*y_stdev)])
    if '--show' in sys.argv:
        fig.show(config=pltconf)

    fig.write_html(f"./full_plot_{material}_{direct}.html")
    if '--pitstop' in sys.argv:
        input()
    # %%
    # Polar of selected band
    print("Band polar plotting...")
    fig_polar = pgo.Figure()
    fig_polar.layout.template = plotly_global
    print(ixs_all)

    tmp = simple_compnts(sols[0], 2)[ixs_all[0]]

    traces = [
        {
            'name': f'{direct} band {round(tmp[nr], 2)}',
            'type': 'scatterpolar',
            'mode': 'markers',
            'marker': {
                'size': 5,
            },
            'theta': [theta for theta in range(0, 370, 10)],
            'r': [abs(simple_compnts(sols[n], 0)[ixs_all[n][nr]] *
                  simple_compnts(sols[n], 1)[ixs_all[n][nr]])
                  for n in range(len(sols))]
        } for nr in range(len(min(ixs_all, key=lambda arr: len(arr))))
    ]
    # adding traces to global storage (for later use)
    config[material]['polar'] += traces
    for tr in traces:
        if 'VV' in tr['name']:
            col_name = tr['name'].replace(' band ', '')
            print(col_name)
            df_rad[col_name] = tr['r']
        if 'VH' in tr['name']:
            col_name = tr['name'].replace(' band ', '')
            print(col_name)
            df_rad[col_name] = tr['r']

    layout = {
        'title': 'Polar intensity of peaks (not normalised)',
        'showlegend': True
    }

    pltconf = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    }

    fig_polar.add_traces(traces)
    fig_polar.update_layout(layout)
    if '--show' in sys.argv:
        fig_polar.show(config=pltconf)

    fig_polar.write_html(f"./polar_{material}_{direct}.html")
df_rad.to_csv(f"rad_{material}.csv", index=False)
# %%
if material == 'si':
    print("Intensity polar Si")
    fig_polar = pgo.Figure()
    fig_polar.layout.template = plotly_global
    rads = [np.zeros(37), np.zeros(37)]
    for tr in config[material]['polar']:
        if 'VV' in tr['name']:
            rads[0] += np.array(tr['r'])
        elif 'VH' in tr['name']:
            rads[1] += np.array(tr['r'])

    traces = [
        {
            'name': f"{direct} intensity",
            'type': 'scatterpolar',
            'mode': 'markers',
            'marker': {
                'size': 5
            },
            'theta': [theta for theta in range(0, 370, 10)],
            'r': rads[nd]
        } for nd, direct in enumerate(directions)
    ]

    traces += [
        {
            'name': 'Summed intensity',
            'type': 'scatterpolar',
            'mode': 'markers',
            'marker': {
                'size': 5,
                'symbol': 'x'
            },
            'theta': [theta for theta in range(0, 370, 10)],
            'r': rads[0]+rads[1]
        }
    ]

    layout = {
        'title': 'Polar intensity (not normalised), VV and VH',
        'showlegend': True
    }

    pltconf = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    }

    fig_polar.add_traces(traces)
    fig_polar.update_layout(layout)
    if '--show' in sys.argv:
        fig_polar.show(config=pltconf)

    fig_polar.write_html(f"./polar_{material}_full.html")
print("All done.")
# %%
