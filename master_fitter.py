# %%
# Imports and prerequisites
import os
import re
import warnings

import plotly.express as pex
import plotly.subplots as psp
import plotly.graph_objects as pgo

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from nano.table import table
from nano.addfun import gdev, lorentz, residual, spectrum
from nano.functions import d, nearest_val, pearson, smoothing, xpeak

warnings.filterwarnings("ignore")
plt.style.use('dark_background')
gav = smoothing


def glob_style(clr): return np.array([1, 1, 1, 1])[:len(clr)]-np.array(clr)


def percentage(u, v):
    val = 100*u/v
    dig = int(abs(np.log10(val)-3))
    return f"{round(val, dig)}%"


# %%
# Configuration
config = {
    'hbn': {
        'init': 5,
        'count': 20,
        'day': '190725',
        'measure': 'polar_hbn',
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
    gd = (gdev(r, M)**2 + gav(r, M)**2)
    resgd = residual(raman, x_av, y, 1/gd)
    p = xpeak(x_av, gd, max(gd), max(gd)/2)
    hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
    Amp = r[p[1]]
    x0 = (x_av[p[0]]+x_av[p[-1]])/2
    v = [abs(Amp), hmhw, x0]
    sol = list(sol)
    sol += [Amp, hmhw, x0]
    sol, h = leastsq(dif, sol)
    r = dif(sol)
    print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

M = config[material]['M']['final']
while len(sol)/3 < config[material]['count']:
    gd = (gdev(r, M)**2 + gav(r, M)**2)
    resgd = residual(raman, x_av, y, 1/gd)
    p = xpeak(x_av, gd, max(gd), max(gd)/2)
    hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
    Amp = r[p[1]]
    x0 = (x_av[p[0]]+x_av[p[-1]])/2
    v = [abs(Amp), hmhw, x0]
    sol = list(sol)
    sol += [Amp, hmhw, x0]
    sol, h = leastsq(dif, sol)
    r = dif(sol)
    print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")


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

    # Only those which meet the requirement of Z >= 0.05
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

# %%
# Main fitter plotter
K = M
gd = (gdev(r, M)**2 + gav(r, M)**2)
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
            'color': 'white'
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
        'text': f'A:{sol[n]}\nw:{sol[n+1]}\nx:{sol[n+2]}',
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 0.5,
            'color': 'white'
        },
        'x': x_av,
        'y': lorentz(sol[n:n+3], x_av)
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

fig.show()

# %%
# Proper logic (fitting all data sets)
for dset in tbl.keys():
    # Attempt to apply model to all data sets
    # dset = '#0' its the same '000'
    y = np.array(tbl[dset]['#Intensity'])
    sol, h = leastsq(residual(raman, x_av, y),
                     config[material]['average'][direct])
    dif = residual(raman, x_av, y)
    print(f"For {dset} deg.: {dif(sol).dot(dif(sol))}")

    ix = 0
    # Main resgd fitter
    M = config[material]['M']['init']
    while len(sol)/3 < config[material]['init']:
        gd = (gdev(r, M)**2 + gav(r, M)**2)
        resgd = residual(raman, x_av, y, 1/gd)
        p = xpeak(x_av, gd, max(gd), max(gd)/2)
        hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
        Amp = r[p[1]]
        x0 = (x_av[p[0]]+x_av[p[-1]])/2
        v = [abs(Amp), hmhw, x0]
        sol = list(sol)
        sol += [Amp, hmhw, x0]
        sol, h = leastsq(dif, sol)
        r = dif(sol)
        print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

    M = config[material]['M']['final']
    while len(sol)/3 < config[material]['count']:
        gd = (gdev(r, M)**2 + gav(r, M)**2)
        resgd = residual(raman, x_av, y, 1/gd)
        p = xpeak(x_av, gd, max(gd), max(gd)/2)
        hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
        Amp = r[p[1]]
        x0 = (x_av[p[0]]+x_av[p[-1]])/2
        v = [abs(Amp), hmhw, x0]
        sol = list(sol)
        sol += [Amp, hmhw, x0]
        sol, h = leastsq(dif, sol)
        r = dif(sol)
        print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

        # # %%
        # Active vs overall surface

        surface = []
        active = []
        for n in range(0, len(sol)-1, 3):
            surface += [np.log10(abs(sol[n]*sol[n+1]))]
            active += [np.log10(abs(sum(lorentz(sol[n:n+3], x_av)*d(x_av))))]
        surface = np.array(surface)
        active = np.array(active)

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

        # Only those which meet the requirement of Z >= 0.05
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


# %%
# Chosen peak plotter

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


sols = config[material]['solutions']['VH']

peak_nr = 36

# Peak selection
k = 10
band = 1540
amp = av_param(band, np.abs(compnts(sols, 0)[peak_nr]), compnts(sols, 2)[peak_nr], k)
w = av_param(band, np.abs(compnts(sols, 1)[peak_nr]), compnts(sols, 2)[peak_nr], k)
x0 = av_param(band, np.abs(compnts(sols, 2)[peak_nr]), compnts(sols, 2)[peak_nr], k)
sol_av = [amp, w, x0]

sol = sols[peak_nr]
y = np.array(tbl[f'#{peak_nr}']['#Intensity'])

# Create figure
fig = pgo.Figure()
fig.layout.template = plotly_global

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
            'x': x_av,
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
            'x': x_av,
            'y': raman(sol, x_av)
        },
        {
            'name': 'choice',
            'type': 'scatter',
            'mode': 'lines',
            'line': {
                'width': 1,
                'color': 'yellow'
            },
            'visible': False,
            'x': x_av,
            'y': lorentz(sol_av, x_av)
        },
        {
            'name': 'choice curve',
            'type': 'scatter',
            'mode': 'lines',
            'line': {
                'width': 1,
                'color': 'magenta'
            },
            'visible': False,
            'x': x_av,
            'y': 1/k * gauss((x_av - band)/k) * 50000
        }
    ]

    for nr, s in enumerate([sol[n:n+3] for n in range(0, len(sol)-1, 3)]):
        traces += [
            {
                'visible': False,
                'text': f'A:{s[0]}\nw:{s[1]}\nx:{s[2]}',
                'line': {
                    'width': 1,
                    'color': 'white'
                },
                'name': f'Comp. #{nr}',
                'x': x_av,
                'y': lorentz(s, x_av)
            }
        ]
    for trace in traces:
        fig.add_trace(trace)
print(len(fig.data))
# Make some traces visible
for n in range(19):
    fig.data[n].visible = True

# Create and add slider
steps = []
for i in range(0, int(len(fig.data)/19), 1):
    step = {
        'method': 'update',
        'args': [
            {'visible': [False]*len(fig.data)},
            {'title': f'Angle {i*10} deg.'}
        ]
    }
    for m in range(i*19, (i+1)*19, 1):
        step["args"][0]["visible"][m] = True
    steps.append(step)

sliders = [{
    'active': 0,
    'currentvalue': {},
    'pad': {'t': 10},
    'steps': steps
}]

fig.update_layout(
    sliders=sliders
)

fig.show()

# %%
fig_polar = pgo.Figure()
fig_polar.layout.template = plotly_global

k = 10
band = 1540
sol_band = []
for n in range(37):
    y = np.array(tbl[f'#{n}']['#Intensity'])
    # amp = av_param(band, np.abs(compnts(sols, 0)[n]), compnts(sols, 2)[n], k)
    # w = av_param(band, np.abs(compnts(sols, 1)[n]), compnts(sols, 2)[n], k)
    # x0 = av_param(band, np.abs(compnts(sols, 2)[n]), compnts(sols, 2)[n], k)
    amp = y[next(nearest_val(x_av, 1582))]-10000
    w = 5
    x0 = 1593.8
    sol_band.append([amp, w, x0])

traces = [
    {
        'name': 'intensity',
        'type': 'scatterpolar',
        'mode': 'markers',
        'marker': {
            'size': 5,
        },
        'theta': [theta for theta in range(0, 370, 10)],
        'r': [sol_band[n][0]*sol_band[n][1]-51000 for n in range(37)]
    }
]

fig_polar.add_traces(traces)

fig_polar.show()
