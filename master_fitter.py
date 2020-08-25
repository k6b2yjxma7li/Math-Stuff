# %%
# Imports and prerequisites
import os
import re
import warnings

import plotly.express as pex
import plotly.graph_objects as pgo
import plotly.subplots as psp

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from scipy.optimize import leastsq

from nano.functions import d, nearest_val, pearson, smoothing, xpeak
from nano.table import table
from nano.addfun import gdev, lorentz, residual, spectrum

warnings.filterwarnings("ignore")
plt.style.use('dark_background')
gav = smoothing


def glob_style(clr): return np.array([1, 1, 1, 1])[:len(clr)]-np.array(clr)


def percentage(u, v):
    val = 100*u/v
    dig = int(abs(np.log10(val)-3))
    # print(dig)
    return f"{round(val, dig)}%"


# %%
# Configuration
config = {
    'hbn': {
        'init': 5,
        'count': 20,
        'day': '190725',
        'measure': 'polar_hbn',
        # 'direct': 'VH',
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
        }
    },
    'grf': {
        'init': 5,
        'count': 15,
        'day': '190726',
        'measure': 'polar_grf',
        # 'direct': 'VH',
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
        }
    },
    'figsave': False
}

# %%
# Data loading and preparations
material = 'grf'
direct = 'VH'

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
        tdict.update({int(dataset[0]): dataset[0]})
for nr, name in enumerate(datafiles):
    name = os.path.join(path, name)
    tbl += {tdict[10*nr]: table().read_csv(open(name, 'r'), delim=r",")}
datafiles = dict(zip(tdict.keys(), datafiles))
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
# %%
# Main resgd fitter
# for M in [10, 20, 30, 50, 75, 100, 150, 200, 250]:
# print(f"M = {M}")
raman = spectrum(lorentz, 3)
dif = residual(raman, x_av, y_av)
ressq = residual(raman, x_av, y_av, 1/y_av**0.5)
res = residual(raman, x_av, y_av, y_stdev)
sol = []

y = y_av
initial = True

r = dif([0, 1, 1])


def resg_gen(fn, u, v, w):
    def _res_(slt):
        return residual(fn, u, v, w)(slt)*np.array(slt).dot(np.array(slt))
    return _res_


M = config[material]['M']['init']
while len(sol)/3 < config[material]['init']:
    gd = (gdev(r, M)**2 + gav(r, M)**2)  # V2
    # resgd = resg_gen(raman, x_av, y, 1/gd)
    resgd = residual(raman, x_av, y, 1/gd)
    p = xpeak(x_av, gd, max(gd), max(gd)/2)
    hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
    Amp = r[p[1]]
    x0 = (x_av[p[0]]+x_av[p[-1]])/2
    v = [abs(Amp), hmhw, x0]
    sol = list(sol)
    sol += [Amp, hmhw, x0]
    # if len(sol)/3 % 3 == 0 or curve_count - len(sol)/3 < 3:
    sol, h = leastsq(dif, sol)
    r = dif(sol)
    # print(f"{int(len(sol)/3)}: {r.dot(r)}")
    print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

M = config[material]['M']['final']
while len(sol)/3 < config[material]['count']:
    gd = (gdev(r, M)**2 + gav(r, M)**2)  # V2
    # resgd = resg_gen(raman, x_av, y, 1/gd)
    resgd = residual(raman, x_av, y, 1/gd)
    p = xpeak(x_av, gd, max(gd), max(gd)/2)
    hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
    Amp = r[p[1]]
    x0 = (x_av[p[0]]+x_av[p[-1]])/2
    v = [abs(Amp), hmhw, x0]
    sol = list(sol)
    sol += [Amp, hmhw, x0]
    # if len(sol)/3 % 3 == 0 or curve_count - len(sol)/3 < 3:
    sol, h = leastsq(dif, sol)
    r = dif(sol)
    # print(f"{int(len(sol)/3)}: {r.dot(r)}")
    print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

# config[material]['average'][direct] = sol.copy()

# %%
# Active vs overall surface
ix = 0
while True:
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    ax.set_title(f"Curve selector for average model")
    ax.set_xlabel('Surface (log10)')
    ax.set_ylabel('Active surface (log10)')
    surface = []
    active = []
    for n in range(0, len(sol)-1, 3):
        surface += [np.log10(abs(sol[n]*sol[n+1]))]
        active += [np.log10(abs(sum(lorentz(sol[n:n+3], x_av)*d(x_av))))]
        ax.text(surface[-1], active[-1], f"{int(n/3)}", ha='left')
    surface = np.array(surface)
    active = np.array(active)
    ax.plot(surface, active, '.', color=glob_style([0, 0, 0]))
    ax.set_aspect(aspect='equal', adjustable='box')

    def bivariate(u, v, r):
        return np.exp(-(u**2 - 2*r*u*v + v**2)/(2*(1-r**2)**0.5))

    def single_density(u, v):
        data = 1/(2*np.pi)*np.exp(-(u**2 + v**2)/2)
        return data

    def set_density(x_p, y_p, x_std=1, y_std=1):
        if '__len__' not in dir(x_std):
            x_std = np.linspace(x_std, x_std, len(x_p))
        if '__len__' not in dir(y_std):
            y_std = np.linspace(y_std, y_std, len(y_p))

        def DENS(u, v):
            data = 0
            for n in range(len(x_p)):
                data += single_density((u-x_p[n])/x_std[n],
                                       (v-y_p[n])/y_std[n])/(x_std[n]*y_std[n])
            return np.array(data)
        return DENS

    sgm = 0.5
    density = set_density(surface, active, sgm, sgm)

    def density_estimator(u):
        D = density(surface, active)
        D /= sum(D)
        return u.dot(D)

    est = np.mean

    X, Y = np.meshgrid(np.linspace(min(surface)-1, max(surface)+1, 1000),
                       np.linspace(min(active)-1, max(active)+1, 1000))

    R = pearson(surface, active, estimator=est)
    s_std = (est(surface**2) - est(surface)**2)**0.5
    a_std = (est(active**2) - est(active)**2)**0.5

    s_m = est(surface)
    a_m = est(active)

    Z = bivariate((X-s_m)/s_std, (Y-a_m)/a_std, R)

    requirement = bivariate((surface-s_m)/s_std, (active-a_m)/a_std, R) >= 0.05

    for n in range(len(surface)):
        if not requirement[n]:
            ax.plot(surface[n], active[n], '.', color='red')

    ax.plot(s_m, a_m, '+', ms=5, color=[1, 0, 1])
    ax.text(s_m, a_m, f"({round(s_m, 2)},{round(a_m, 2)})", color=[1, 1, 0],
            fontsize=8)
    levs = np.array([1e-4, 2e-4, 5e-4, 0.001, 0.002, 0.005, 0.01,
                     0.02, 0.05, 0.1, 0.2, 0.5, 1])*max(Z.flatten())
    cs = ax.contour(X, Y, Z, linewidths=0.7, levels=levs)
    ax.clabel(cs, inline=True, fontsize=8)

    # # %%
    # Only those which meet the requirement of Z >= 0.05
    new_req = []
    for n in range(len(requirement)):
        for k in range(3):
            new_req += [requirement[n]]
        if not requirement[n]:
            print(f"Trimmed: index {n}")
    new_req = np.array(new_req)
    sol = sol[new_req]
    # fig.savefig(f"./.data/{direct}/selector/average_selector_{direct}_{ix}.pdf")
    if False not in requirement:
        print("No trimming")
        break
    ix += 1
# setting number of curves
# if initial:
#     solutions += [sol]
#     config[material]['average'][direct] = sol
#     curve_count = int(len(sol)/3)
#     initial = False
# config[material]['average'][direct] = sol

config[material]['average'][direct] = sol.copy()

# %%
# Main fitter plotter
K = M
gd = (gdev(r, M)**2 + gav(r, M)**2)  # V2
# res = residual(raman, x_av, y, y_stdev)
r = dif(sol)

plotly_global = 'plotly_dark'
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
        'x': x_av,
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
        'x': x_av,
        'y': raman(sol, x_av)
    },
    {
        'type': 'scatter',
        'mode': 'markers',
        'name': 'data stdev',
        'marker': {
            'size': 1,
            'color': 'yellow'
        },
        'x': x_av,
        'y': y_stdev
    },
    {
        'type': 'scatter',
        'mode': 'lines',
        'name': 'penalty',
        'line': {
            'width': 0.1,
            'color': 'white',
            # 'dash': 'dash'
        },
        'yaxis': 'y2',
        'x': x_av,
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
        'x': x_av,
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
        'x': x_av,
        'y': np.linspace(np.mean(r), np.mean(r), len(x_av))
    },
    {
        'type': 'scatter',
        'mode': 'lines',
        'name': 'residual average (full)',
        'line': {
            'width': 1,
            'color': 'yellow'
        },
        'x': x_av,
        'y': np.linspace(np.std(r), np.std(r), len(x_av))
    },
    {
        'type': 'scatter',
        'mode': 'lines',
        'name': 'residual average',
        'line': {
            'width': 0.2,
            'color': 'white'
        },
        'x': x_av,
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
        'x': x_av,
        'y': gdev(r, K)
    },
]

layout = {
    # 'showlegend': False,
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
    # 'yaxis2_title': 'penalty',
    'scene1': {
        # 'name': 'Data fit with penalty',
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

# for nr, traces in enumerate([traces1, traces2]):
for trace in traces1:
    if 'yaxis' in trace:
        fig.add_trace(trace, row=1, col=1, secondary_y=True)
    else:
        fig.add_trace(trace, row=1, col=1, secondary_y=False)

for trace in traces2:
    if 'yaxis' in trace:
        fig.add_trace(trace, row=2, col=1)
    else:
        fig.add_trace(trace, row=2, col=1)
fig.update_layout(layout)

fig.show()

# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
# ax[0].set_title(f"Average data with stddev")
# ax[1].set_title("Residual with average and deviations")
# ax[0].plot(x_av, y, '.', ms=0.7, lw=0.7, color=glob_style([0, 0, 0]))
# ax[0].plot(x_av, y_stdev, '.', ms=0.7, lw=0.7, color=glob_style([0, 0, 1]))

# ax1 = plt.twinx(ax[0])
# # ax1.set_ylim([0, 1200])

# ax[1].plot(x_av, r, '.', lw=0.7, ms=0.8, color=glob_style([0, 0, 0]))

# ax2 = plt.twinx(ax[1])
# ax2.set_ylim([0, 1200])
# ax2.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7, color=glob_style([0, 0, 0]))
# ax2.plot(x_av, np.linspace(np.std(r), np.std(r), len(x_av)), '--', lw=0.7,
#          ms=0.7, color=glob_style([0, 0, 1]))

# ax[1].plot(x_av, np.linspace(np.mean(r), np.mean(r), len(x_av)), '-.', lw=0.7,
#            ms=0.7, color=glob_style([0, 0, 1]))
# ax[1].plot(x_av, smoothing(r, K), '--', lw=0.7, ms=0.7,
#            color=glob_style([0, 0, 0]))

# ax1.plot(x_av, gd, '-', lw=0.7, ms=0.7, color=glob_style([0, 0, 0]))

# ax[0].plot(x_av, lorentz(v, x_av), '-', color=glob_style([0, 0, 0, 0.3]),
#            lw=0.9, ms=0.9)
# ax[0].plot(x_av, raman(sol, x_av), '-', color=glob_style([1, 0, 0]),
#            lw=0.9)
# ax[0].plot(x_av, raman(sol, x_av)+smoothing(r, K), '--',
#            color=glob_style([1, 0, 0]), lw=0.9)

# fig.savefig(f"./.data/{direct}/fit/average_{direct}_fit.pdf")

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
        'x': x_av,
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
        'x': x_av,
        'y': raman(sol, x_av)
    }
]

spectrals = []

for nr, n in enumerate(range(0, len(sol)-1, 3)):
    spectrals.append({
        'name': f'Comp. #{nr}',
        'text': f'A:{sol[n]}\nx:{sol[n+1]}\nw:{sol[n+2]}',
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'white'
        },
        # 'x': np.linspace(min(x_av), max(x_av), len(x_av)*3),
        'x': x_av,
        # 'y': lorentz(sol[n:n+3], np.linspace(min(x_av), max(x_av),
        #  len(x_av)*3))
        'y': lorentz(sol[n:n+3], x_av)
    })

traces += spectrals

layout = {
    'title': 'Spectral components for average model',
    # 'legend': {
    #     'orientation': 'h',
    #     'yanchor': 'bottom',
    #     'y': 1.02,
    #     'xanchor': 'right',
    #     'x': 1
    # },
    'xaxis_title': 'wavenumber [cm^-1]',
    'yaxis_title': 'counts (data + fit + compnts)',
    # 'yaxis2_title': 'penalty',
    'scene': {
        # 'name': 'Data fit with penalty',
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

fig.show()

# # %%
# Visual of spectrum components
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_title(f"Spectral components for average model")
# ax.set_xlim([min(x_av), max(x_av)])
# ax.set_ylim([0, max(y)*1.25])
# ax.plot(x_av, y, '.', ms=2, color=glob_style([0, 0, 0]))
# ax.plot(x_av, raman(sol, x_av), '-', lw=0.7, color=glob_style([1, 0, 0]))
# for n in range(0, len(sol)-1, 3):
#     ax.plot(x_av, lorentz(sol[n:n+3], x_av), lw=0.7,
#             color=glob_style([0, 0, 0, 0.3]))
#     if 500 < sol[n:n+3][2] < 3000:
#         if 0 < abs(sol[n:n+3][0]) < 14000:
#             ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2,
#                     color=glob_style([1, 0, 0]))
#             ax.text(sol[n:n+3][2], abs(sol[n:n+3][0]),
#                     f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][2]),1))}")
#         else:
#             ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2,
#                     color=glob_style([1, 0, 0]))
#             ax.text(sol[n:n+3][2], 14000,
#                     f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][2]),1))}")
# fig.savefig(f"./.data/{direct}/comp/average_{direct}_comp.pdf")

# %%
# Proper logic
for dset in tbl.keys():
    # # %%
    # Attempt to apply model to data set
    # fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
    # dset = '#0' its the same '000'
    y = np.array(tbl[dset]['#Intensity'])
    sol, h = leastsq(residual(raman, x_av, y),
                     config[material]['average'][direct])
    dif = residual(raman, x_av, y)
    print(f"For {dset} deg.: {dif(sol).dot(dif(sol))}")

    ix = 0
    # # %%
    # Main resgd fitter
    M = config[material]['M']['init']
    while len(sol)/3 < config[material]['init']:
        gd = (gdev(r, M)**2 + gav(r, M)**2)  # V2
        # resgd = resg_gen(raman, x_av, y, 1/gd)
        resgd = residual(raman, x_av, y, 1/gd)
        p = xpeak(x_av, gd, max(gd), max(gd)/2)
        hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
        Amp = r[p[1]]
        x0 = (x_av[p[0]]+x_av[p[-1]])/2
        v = [abs(Amp), hmhw, x0]
        sol = list(sol)
        sol += [Amp, hmhw, x0]
        # if len(sol)/3 % 3 == 0 or curve_count - len(sol)/3 < 3:
        sol, h = leastsq(dif, sol)
        r = dif(sol)
        # print(f"{int(len(sol)/3)}: {r.dot(r)}")
        print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

    M = config[material]['M']['final']
    while len(sol)/3 < config[material]['count']:
        gd = (gdev(r, M)**2 + gav(r, M)**2)  # V2
        # resgd = resg_gen(raman, x_av, y, 1/gd)
        resgd = residual(raman, x_av, y, 1/gd)
        p = xpeak(x_av, gd, max(gd), max(gd)/2)
        hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
        Amp = r[p[1]]
        x0 = (x_av[p[0]]+x_av[p[-1]])/2
        v = [abs(Amp), hmhw, x0]
        sol = list(sol)
        sol += [Amp, hmhw, x0]
        # if len(sol)/3 % 3 == 0 or curve_count - len(sol)/3 < 3:
        sol, h = leastsq(dif, sol)
        r = dif(sol)
        # print(f"{int(len(sol)/3)}: {r.dot(r)}")
        print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

        # # %%
        # Active vs overall surface

        # fig, ax = plt.subplots()
        # fig.set_size_inches(15, 15)
        # ax.set_title(f"Curve selector for {dset} degrees {direct}")
        # ax.set_xlabel('Surface (log10)')
        # ax.set_ylabel('Active surface (log10)')
        surface = []
        active = []
        for n in range(0, len(sol)-1, 3):
            surface += [np.log10(abs(sol[n]*sol[n+1]))]
            active += [np.log10(abs(sum(lorentz(sol[n:n+3], x_av)*d(x_av))))]
            # ax.text(surface[-1], active[-1], f"{int(n/3)}", ha='left')
        surface = np.array(surface)
        active = np.array(active)
        # ax.plot(surface, active, '.', color=glob_style([0, 0, 0]))
        # ax.set_aspect(aspect='equal', adjustable='box')

        def bivariate(u, v, r):
            return np.exp(-(u**2 - 2*r*u*v + v**2)/(2*(1-r**2)**0.5))

        def single_density(u, v):
            data = 1/(2*np.pi)*np.exp(-(u**2 + v**2)/2)
            return data

        def set_density(x_p, y_p, x_std=1, y_std=1):
            if '__len__' not in dir(x_std):
                x_std = np.linspace(x_std, x_std, len(x_p))
            if '__len__' not in dir(y_std):
                y_std = np.linspace(y_std, y_std, len(y_p))

            def DENS(u, v):
                data = 0
                for n in range(len(x_p)):
                    data += (single_density((u-x_p[n])/x_std[n],
                             (v-y_p[n])/y_std[n])/(x_std[n]*y_std[n]))
                return np.array(data)
            return DENS

        sgm = 0.1
        density = set_density(surface, active, sgm, sgm)

        def density_estimator(u):
            D = density(surface, active)
            D /= sum(D)
            return u.dot(D)

        est = np.mean

        R = pearson(surface, active, estimator=est)
        s_std = (est(surface**2) - est(surface)**2)**0.5
        a_std = (est(active**2) - est(active)**2)**0.5

        s_m = est(surface)
        a_m = est(active)

        X, Y = np.meshgrid(np.linspace(min(surface)-1, max(surface)+1, 1000),
                           np.linspace(min(active)-1, max(active)+1, 1000))

        Z = bivariate((X-s_m)/s_std, (Y-a_m)/a_std, R)

        requirement = bivariate((surface-s_m)/s_std,
                                (active-a_m)/a_std, R) >= 0.05

        # for n in range(len(surface)):
        #     if not requirement[n]:
        #         ax.plot(surface[n], active[n], '.', color='red')

        # ax.plot(s_m, a_m, '+', ms=5, color=[1, 0, 1])
        # ax.text(s_m, a_m, f"({round(s_m, 2)},{round(a_m, 2)})",
        #         color=[1, 1, 0], fontsize=8)
        # levs = np.array([1e-4, 2e-4, 5e-4, 0.001, 0.002, 0.005, 0.01, 0.02,
        #                  0.05, 0.1, 0.2, 0.5, 1])*max(Z.flatten())
        # cs = ax.contour(X, Y, Z, linewidths=0.7, levels=levs)
        # ax.clabel(cs, inline=True, fontsize=8)

        # # %%
        # Only those which meet the requirement of Z >= 0.05
        new_req = []
        for n in range(len(requirement)):
            for k in range(3):
                new_req += [requirement[n]]
            if not requirement[n]:
                print(f"Trimmed: index {n}")
        new_req = np.array(new_req)
        sol = sol[new_req]
        # fig.savefig(f"./.data/{direct}/selector/average_selector_{direct}_{ix}.pdf")
        if False not in requirement:
            print("No trimming")
            break
        ix += 1
    # setting number of curves
    # if initial:
    #     solutions += [sol]
    #     config[material]['average'][direct] = sol
    #     curve_count = int(len(sol)/3)
    #     initial = False
    config[material]['solutions'][direct].append(sol)

    # # %%
    # # Main fitter plotter
    # K = M
    # res = residual(raman, x_av, y, y_stdev)
    # r = dif(sol)
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
    # ax[0].set_title(f"Data with stddev for {dset} degrees {direct}")
    # ax[1].set_title("Residual with average and deviations")
    # ax[0].plot(x_av, y, '.', ms=0.7, lw=0.7, color=glob_style([0, 0, 0]))
    # ax[0].plot(x_av, y_stdev, '.', ms=0.7, lw=0.7, color=glob_style([0, 0, 1]))

    # ax1 = plt.twinx(ax[0])
    # ax1.set_ylim([0, 1200])

    # ax[1].plot(x_av, r, '.', lw=0.7, ms=0.8, color=glob_style([0, 0, 0]))

    # ax2 = plt.twinx(ax[1])
    # ax2.set_ylim([0, 1200])
    # ax2.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7,
    #          color=glob_style([0, 0, 0]))
    # ax2.plot(x_av, np.linspace(np.std(r), np.std(r), len(x_av)), '--', lw=0.7,
    #          ms=0.7, color=glob_style([0, 0, 1]))

    # ax[1].plot(x_av, np.linspace(np.mean(r), np.mean(r), len(x_av)), '-.',
    #            lw=0.7, ms=0.7, color=glob_style([0, 0, 1]))
    # ax[1].plot(x_av, smoothing(r, K), '--', lw=0.7, ms=0.7,
    #            color=glob_style([0, 0, 0]))

    # ax1.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7,
    #          color=glob_style([0, 0, 0]))

    # ax[0].plot(x_av, lorentz(v, x_av), '-', color=glob_style([0, 0, 0, 0.3]),
    #            lw=0.9, ms=0.9)
    # ax[0].plot(x_av, raman(sol, x_av), '-', color=glob_style([1, 0, 0]),
    #            lw=0.9)
    # ax[0].plot(x_av, raman(sol, x_av)+smoothing(r, K), '--',
    #            color=glob_style([1, 0, 0]), lw=0.9)

    # fig.savefig(f"./.data/{direct}/fit/{dset}_{direct}_fit.pdf")
    # # # %%
    # # Visual of spectrum components
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_title(f"Spectral components for {dset} degrees {direct}")
    # ax.set_xlim([min(x_av), max(x_av)])
    # ax.set_ylim([0, max(y)*1.25])
    # ax.plot(x_av, y, '.', ms=2, color=glob_style([0, 0, 0]))
    # ax.plot(x_av, raman(sol, x_av), '-', lw=0.7, color=glob_style([1, 0, 0]))
    # for n in range(0, len(sol)-1, 3):
    #     ax.plot(x_av, lorentz(sol[n:n+3], x_av), lw=0.7,
    #             color=glob_style([0, 0, 0, 0.3]))
    #     if 500 < sol[n:n+3][2] < 3000:
    #         if 0 < abs(sol[n:n+3][0]) < 14000:
    #             ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2,
    #                     color=glob_style([1, 0, 0]))
    #             ax.text(sol[n:n+3][2], abs(sol[n:n+3][0]),
    #                     f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][2]),1))}")
    #         else:
    #             ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2,
    #                     color=glob_style([1, 0, 0]))
    #             ax.text(sol[n:n+3][2], 14000,
    #                     f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][2]),1))}")
    # # fig.savefig(f"./.data/{direct}/comp/{dset}_{direct}_comp.pdf")
    # plt.show()
# %%
# Radial for chosen peak
# fig, ax = plt.subplots(figsize=(10, 10))

# sols = solutions[:]

sols = config[material]['solutions']['VH']

fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(221)
ax1.set_title('Curves of choice')
ax2 = fig.add_subplot(222)
ax2.set_title('Peak position')
ax3 = fig.add_subplot(223)
ax3.set_title('FWHM')
ax4 = fig.add_subplot(224, projection='polar')
ax4.set_title('Amplitude')
# ax = fig.add_subplot(121, projection='polar')
# ax1.set_title('Curves of choice')
# fig1 = plt.figure(figsize=(10, 10))
max_y = 0
fltr = 2
comp = 2
exclusion = np.array(np.linspace(0, 0, len(sols)), dtype=bool)
addressing = np.array(np.linspace(-1, -1, len(sols)), dtype=int)
# exclusion[21] = True
for nr, s in enumerate(sols):
    if not exclusion[nr]:
        s = np.array(s)
        ix = next(nearest_val(s[np.arange(0, len(s), 1) % 3 == fltr], 1367))
        if addressing[nr] != -1:
            ix = addressing[nr]
        # print(ix)
        # curr_peak = abs(s[0+3*ix])*abs(s[1+3*ix])
        ax1.plot(x_av, lorentz(s[3*ix:3*ix+3], x_av), '-', lw=0.7)
        ax1.text(x_av[-1], lorentz(s[3*ix:3*ix+3], x_av[-1]), f"{nr}/{ix}")
        pos = abs(s[2+3*ix])
        hmhw = abs(s[1+3*ix])
        # amp = abs(s[0+3*ix])*hmhw
        amp = abs(s[0+3*ix])
        ax2.plot(nr*10, pos, '.', color=[0, 1, 1], ms=5)
        ax3.plot(nr*10, hmhw, '.', color=[0, 1, 1], ms=5)
        ax4.plot(np.pi*nr/18, amp, '.', color=[0, 1, 1], ms=5)

        # ax1.text(s[2+3*ix], lorentz(s[3*ix:3*ix+3], s[2+3*ix]), f"{nr}/{ix}")
        if amp > max_y:
            max_y = amp*1.1
ax4.set_ylim([0, max_y])
# ax.set_ylim([0, 2000])

# %%
# %%
fig = pex.scatter()
fig.layout.template = 'plotly_dark'
gd = (gdev(r, M)**2 + gav(r, M)**2)  # V2
mat_av_data = config[material]['average'][direct]
traces = [
    {
        'x': x_av,
        'y': residual(raman, x_av, y_av, 1/gd)(mat_av_data),
        'name': 'Res1 (penalty 1/gd)',
        'mode': 'markers',
        'marker': {'size': 2},
        'yaxis': 'y1'
    },
    {
        'x': x_av,
        'y': residual(raman, x_av, y_av, gd)(mat_av_data),
        'name': 'Res2 (penalty gd)',
        'mode': 'markers',
        'marker': {'size': 2},
        'yaxis': 'y2'
    }]
layout = {
    'title': 'Residuals',
    'yaxis1': {
        'color': '#5577ff'
    },
    'yaxis2': {
        'overlaying': 'y',
        'side': 'right',
        'color': '#ff7755'
    }
}
fig.add_traces(traces)
fig.update_layout(layout)
fig.show()

# %%
fig = pex.scatter()
fig.layout.template = 'plotly_dark'
gd = gdev(y_av, 10000)
traces = [
    {
        'x': x_av,
        'y': y_av,
        'name': 'Data',
        'mode': 'markers',
        'marker': {'size': 2},
        'yaxis': 'y1'
    },
    {
        'x': x_av,
        'y': raman(config[material]['average'][direct], x_av),
        'name': 'Fit',
        'mode': 'lines',
        'line': {'width': 1}
    }
]
layout = {
    'title': 'Data vs fit',
    'yaxis2': {
        'overlaying': 'y',
        'side': 'right'
    }
}
fig.add_traces(traces)
fig.update_layout(layout)
fig.show()


# %%
fig = pex.scatter_3d()

fig.layout.template = 'plotly_dark'

# splitting main vector into sub curves

x_coeff = []
y_coeff = []
z_coeff = []
labels = []

for n in range(0, len(sol)-1, 3):
    A, b, x0 = sol[n:n+3]
    x_coeff += [abs(A)]
    y_coeff += [abs(b)]
    z_coeff += [x0 - np.mean(x_av)]
    labels += [f'{int(n/3)}']


traces = [
    {
        'type': 'scatter3d',
        'x': x_coeff,
        'y': y_coeff,
        'z': z_coeff,
        'mode': 'markers',
        'text': labels,
        'marker': {'size': 2}
    }
]

layout = {
    'scene': {
        'xaxis': {
            'title': 'x: Amplitude'
        },
        'yaxis': {
            'title': 'y: hmhw'
        },
        'zaxis': {
            'title': 'z: x0'
        }
    },
    'margin': {
        't': 25,
        'b': 5,
        'l': 5,
        'r': 5
    }
}

fig.add_traces(traces)
fig.update_layout(layout)

fig.show()

# %%
