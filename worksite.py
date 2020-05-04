# %%
# Imports and prerequisites
import os
import re
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from scipy.optimize import leastsq

from nano.functions import d, nearest_val, pearson, smoothing, xpeak
from nano.table import table
from nano.addfun import gdev, lorentz, residual, spectrum

warnings.filterwarnings("ignore")

glob_style = lambda clr: np.array([1,1,1,1])[:len(clr)]-np.array(clr)
plt.style.use('dark_background')

def percentage(u, v):
    val = 100*u/v
    dig = int(abs(np.log(val)))
    return f"{round(val, dig)}%"


# %%
# Data loading and preparations
direct = "VH"
measure = "polar_hbn"
day = "190725"
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
## %%
sol = []
raman = spectrum(lorentz, 3)
dif = residual(raman, x_av, y_av)
ressq = residual(raman, x_av, y_av, 1/y_av**0.5)
res = residual(raman, x_av, y_av, y_stdev)
M = 100
scale = 1

y = y_av
curve_count = 20
initial = True
solutions = []

# %%
# Main resgd fitter
r = dif([0, 1, 1])
while len(sol)/3 < curve_count:
    gd = gdev(r, M)
    resgd = residual(raman, x_av, y, 1/gd)
    p = xpeak(x_av, gd, max(gd), max(gd)/2)
    hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
    Amp = r[p[1]]
    x0 = (x_av[p[0]]+x_av[p[-1]])/2
    v = [abs(Amp), hmhw, x0]
    sol = list(sol)
    sol += [Amp, hmhw, x0]
    if len(sol)/3 % 3 == 0 or curve_count - len(sol)/3 < 3:
        sol, h = leastsq(resgd, sol)
    r = dif(sol)
    # print(f"{int(len(sol)/3)}: {r.dot(r)}")
    print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")

sola = sol.copy()


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
    ax.plot(surface, active, '.', color=glob_style([0,0,0]))
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

    ax.plot(s_m, a_m, '+', ms=5, color=[1,0,1])
    ax.text(s_m, a_m, f"({round(s_m, 2)},{round(a_m, 2)})", color=[1,1,0], fontsize=8)
    levs = np.array([1e-4, 2e-4, 5e-4, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])*max(Z.flatten())
    cs = ax.contour(X, Y, Z, linewidths=0.7, levels=levs)
    ax.clabel(cs, inline=True, fontsize=8)


    ## %%
    # Only those which meet the requirement of Z >= 0.05
    sol = list(sol)
    for n in range(len(requirement)):
        if not requirement[n]:
            print(f"Trimmed: index {n}")
            #   popping three n-th elements in a row (removing A, hmhw, x0)
            for i in range(3):
                sol.pop(n)
    # fig.savefig(f"./.data/{direct}/selector/average_selector_{direct}_{ix}.pdf")
    if False not in requirement:
        print("No trimming")
        break
    ix += 1
# setting number of curves 
# if initial:
#     solutions += [sol]
#     sola = sol
#     curve_count = int(len(sol)/3)
#     initial = False
sola = sol

# %%
# Main fitter plotter
K = M
res = residual(raman, x_av, y, y_stdev)
r = dif(sol)
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
ax[0].set_title(f"Average data with stddev")
ax[1].set_title("Residual with average and deviations")
ax[0].plot(x_av, y, '.', ms=0.7, lw=0.7, color=glob_style([0,0,0]))
ax[0].plot(x_av, y_stdev, '.', ms=0.7, lw=0.7, color=glob_style([0,0,1]))

ax1 = plt.twinx(ax[0])
ax1.set_ylim([0, 1200])

ax[1].plot(x_av, r, '.', lw=0.7, ms=0.8, color=glob_style([0,0,0]))

ax2 = plt.twinx(ax[1])
ax2.set_ylim([0, 1200])
ax2.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7, color=glob_style([0,0,0]))
ax2.plot(x_av, np.linspace(np.std(r), np.std(r), len(x_av)), '--', lw=0.7, ms=0.7, color=glob_style([0,0,1]))

ax[1].plot(x_av, np.linspace(np.mean(r), np.mean(r), len(x_av)), '-.', lw=0.7, ms=0.7, color=glob_style([0,0,1]))
ax[1].plot(x_av, smoothing(r, K), '--', lw=0.7, ms=0.7, color=glob_style([0,0,0]))

ax1.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7, color=glob_style([0,0,0]))

ax[0].plot(x_av, lorentz(v, x_av), '-', color=glob_style([0,0,0,0.3]), lw=0.9, ms=0.9)
ax[0].plot(x_av, raman(sol, x_av), '-', color=glob_style([1,0,0]), lw=0.9)
ax[0].plot(x_av, raman(sol, x_av)+smoothing(r, K), '--', color=glob_style([1,0,0]), lw=0.9)

# fig.savefig(f"./.data/{direct}/fit/average_{direct}_fit.pdf")
## %%
# Visual of spectrum components
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title(f"Spectral components for average model")
ax.set_xlim([min(x_av), max(x_av)])
ax.set_ylim([0, max(y)*1.25])
ax.plot(x_av, y, '.', ms=2, color=glob_style([0,0,0]))
ax.plot(x_av, raman(sol, x_av), '-', lw=0.7, color=glob_style([1,0,0]))
for n in range(0, len(sol)-1, 3):
    ax.plot(x_av, lorentz(sol[n:n+3], x_av), lw=0.7, color=glob_style([0,0,0,0.3]))
    if 500 < sol[n:n+3][2] < 3000:
        if 0 < abs(sol[n:n+3][0]) < 14000:
            ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2, color=glob_style([1,0,0]))
            ax.text(sol[n:n+3][2], abs(sol[n:n+3][0]), f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][0]),1))}")
        else:
            ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2, color=glob_style([1,0,0]))
            ax.text(sol[n:n+3][2], 14000, f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][0]),1))}")
# fig.savefig(f"./.data/{direct}/comp/average_{direct}_comp.pdf")

# %%
# Proper logic
for dset in tbl.keys():
    ## %%
    # Attempt to apply model to data set
    # fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
    # dset = '#0' its the same '000'
    y = np.array(tbl[dset]['#Intensity'])
    sol, h = leastsq(residual(raman, x_av, y, y**0.5), sola)
    dif = residual(raman, x_av, y)
    print(f"For {dset} deg.: {dif(sol).dot(dif(sol))}")

    ix = 0
    ## %%
    # Main resgd fitter
    while True:
        r = dif(sol)
        while len(sol)/3 < curve_count:
            gd = gdev(r, M)
            resgd = residual(raman, x_av, y, 1/gd)
            p = xpeak(x_av, gd, max(gd), max(gd)/2)
            hmhw = abs(x_av[p[0]] - x_av[p[-1]])/2
            Amp = r[p[1]]
            x0 = (x_av[p[0]]+x_av[p[-1]])/2
            v = [abs(Amp), hmhw, x0]
            sol = list(sol)
            sol += [Amp, hmhw, x0]
            sol, h = leastsq(resgd, sol)
            r = dif(sol)
            print(f"{int(len(sol)/3)}: {percentage(r.dot(r), y.dot(y))}")


        ## %%
        # Active vs overall surface
    
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 15)
        ax.set_title(f"Curve selector for {dset} degrees {direct}")
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
        ax.plot(surface, active, '.', color=glob_style([0,0,0]))
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

        R = pearson(surface, active, estimator=est)
        s_std = (est(surface**2) - est(surface)**2)**0.5
        a_std = (est(active**2) - est(active)**2)**0.5

        s_m = est(surface)
        a_m = est(active)

        X, Y = np.meshgrid(np.linspace(min(surface)-1, max(surface)+1, 1000), 
                           np.linspace(min(active)-1, max(active)+1, 1000))

        Z = bivariate((X-s_m)/s_std, (Y-a_m)/a_std, R)

        requirement = bivariate((surface-s_m)/s_std, (active-a_m)/a_std, R) >= 0.05

        for n in range(len(surface)):
            if not requirement[n]:
                ax.plot(surface[n], active[n], '.', color='red')

        ax.plot(s_m, a_m, '+', ms=5, color=[1,0,1])
        ax.text(s_m, a_m, f"({round(s_m, 2)},{round(a_m, 2)})", color=[1,1,0], fontsize=8)
        levs = np.array([1e-4, 2e-4, 5e-4, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])*max(Z.flatten())
        cs = ax.contour(X, Y, Z, linewidths=0.7, levels=levs)
        ax.clabel(cs, inline=True, fontsize=8)


        ## %%
        # Only those which meet the requirement of Z >= 0.05
        sol = list(sol)
        for n in range(len(requirement)):
            if not requirement[n]:
                print(f"Trimmed: index {n}")
                #   popping three n-th elements in a row (removing A, hmhw, x0)
                for i in range(3):
                    sol.pop(n)
        # fig.savefig(f"./.data/{direct}/selector/{dset}_selector_{direct}_{ix}.pdf")
        ix += 1
        if False not in requirement:
            print("No trimming")
            break
    # setting number of curves 
    # if initial:
    #     solutions += [sol]
    #     sola = sol
    #     curve_count = int(len(sol)/3)
    #     initial = False
    solutions += [sol]

    ## %%
    # Main fitter plotter
    K = 50
    res = residual(raman, x_av, y, y_stdev)
    r = dif(sol)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
    ax[0].set_title(f"Data with stddev for {dset} degrees {direct}")
    ax[1].set_title("Residual with average and deviations")
    ax[0].plot(x_av, y, '.', ms=0.7, lw=0.7, color=glob_style([0,0,0]))
    ax[0].plot(x_av, y_stdev, '.', ms=0.7, lw=0.7, color=glob_style([0,0,1]))

    ax1 = plt.twinx(ax[0])
    ax1.set_ylim([0, 1200])

    ax[1].plot(x_av, r, '.', lw=0.7, ms=0.8, color=glob_style([0,0,0]))

    ax2 = plt.twinx(ax[1])
    ax2.set_ylim([0, 1200])
    ax2.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7, color=glob_style([0,0,0]))
    ax2.plot(x_av, np.linspace(np.std(r), np.std(r), len(x_av)), '--', lw=0.7, ms=0.7, color=glob_style([0,0,1]))

    ax[1].plot(x_av, np.linspace(np.mean(r), np.mean(r), len(x_av)), '-.', lw=0.7, ms=0.7, color=glob_style([0,0,1]))
    ax[1].plot(x_av, smoothing(r, K), '--', lw=0.7, ms=0.7, color=glob_style([0,0,0]))

    ax1.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7, color=glob_style([0,0,0]))

    ax[0].plot(x_av, lorentz(v, x_av), '-', color=glob_style([0,0,0,0.3]), lw=0.9, ms=0.9)
    ax[0].plot(x_av, raman(sol, x_av), '-', color=glob_style([1,0,0]), lw=0.9)
    ax[0].plot(x_av, raman(sol, x_av)+smoothing(r, K), '--', color=glob_style([1,0,0]), lw=0.9)

    # fig.savefig(f"./.data/{direct}/fit/{dset}_{direct}_fit.pdf")
    ## %%
    # Visual of spectrum components
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Spectral components for {dset} degrees {direct}")
    ax.set_xlim([min(x_av), max(x_av)])
    ax.set_ylim([0, max(y)*1.25])
    ax.plot(x_av, y, '.', ms=2, color=glob_style([0,0,0]))
    ax.plot(x_av, raman(sol, x_av), '-', lw=0.7, color=glob_style([1,0,0]))
    for n in range(0, len(sol)-1, 3):
        ax.plot(x_av, lorentz(sol[n:n+3], x_av), lw=0.7, color=glob_style([0,0,0,0.3]))
        if 500 < sol[n:n+3][2] < 3000:
            if 0 < abs(sol[n:n+3][0]) < 14000:
                ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2, color=glob_style([1,0,0]))
                ax.text(sol[n:n+3][2], abs(sol[n:n+3][0]), f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][0]),1))}")
            else:
                ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2, color=glob_style([1,0,0]))
                ax.text(sol[n:n+3][2], 14000, f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][0]),1))}")
    # fig.savefig(f"./.data/{direct}/comp/{dset}_{direct}_comp.pdf")



# %%
dset = "000"
y = tbl[dset]["#Intensity"]
sol = solutions[0]
res = residual(raman, x_av, y, y_stdev)
dif = residual(raman, x_av, y)
r = dif(sol)
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
ax[0].set_title(f"Data with stddev for {dset} degrees {direct}")
ax[1].set_title("Residual with average and deviations")
ax[0].plot(x_av, y, '.', ms=0.7, lw=0.7, color=glob_style([0,0,0]))
ax[0].plot(x_av, y_stdev, '.', ms=0.7, lw=0.7, color=glob_style([0,0,1]))

ax1 = plt.twinx(ax[0])
ax1.set_ylim([0, 1200])

ax[1].plot(x_av, r, '.', lw=0.7, ms=0.8, color=glob_style([0,0,0]))

ax2 = plt.twinx(ax[1])
ax2.set_ylim([0, 1200])
ax2.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7, color=glob_style([0,0,0]))
ax2.plot(x_av, np.linspace(np.std(r), np.std(r), len(x_av)), '--', lw=0.7, ms=0.7, color=glob_style([0,0,1]))

ax[1].plot(x_av, np.linspace(np.mean(r), np.mean(r), len(x_av)), '-.', lw=0.7, ms=0.7, color=glob_style([0,0,1]))
ax[1].plot(x_av, smoothing(r, K), '--', lw=0.7, ms=0.7, color=glob_style([0,0,0]))

ax1.plot(x_av, gdev(r, K), '-', lw=0.7, ms=0.7, color=glob_style([0,0,0]))

ax[0].plot(x_av, lorentz(v, x_av), '-', color=glob_style([0,0,0,0.3]), lw=0.9, ms=0.9)
ax[0].plot(x_av, raman(sol, x_av), '-', color=glob_style([1,0,0]), lw=0.9)
ax[0].plot(x_av, raman(sol, x_av)+smoothing(r, K), '--', color=glob_style([1,0,0]), lw=0.9)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title(f"Spectral components for average model")
ax.set_xlim([min(x_av), max(x_av)])
ax.set_ylim([0, max(y)*1.25])
ax.plot(x_av, y, '.', ms=2, color=glob_style([0,0,0]))
ax.plot(x_av, raman(sol, x_av), '-', lw=0.7, color=glob_style([1,0,0]))
for n in range(0, len(sol)-1, 3):
    ax.plot(x_av, lorentz(sol[n:n+3], x_av), lw=0.7, color=glob_style([0,0,0,0.3]))
    if 500 < sol[n:n+3][2] < 3000:
        if 0 < abs(sol[n:n+3][0]) < 14000:
            ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2, color=glob_style([1,0,0]))
            ax.text(sol[n:n+3][2], abs(sol[n:n+3][0]), f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][0]),1))}")
        else:
            ax.plot(sol[n:n+3][2], abs(sol[n:n+3][0]), '.', ms=2, color=glob_style([1,0,0]))
            ax.text(sol[n:n+3][2], 14000, f"[{int(n/3)}]  {str(round(abs(sol[n:n+3][0]),1))}")
# %%
# Radial for chosen peak
# fig, ax = plt.subplots(figsize=(10, 10))
fig = plt.figure(figsize=(10, 10))
for nr, s in enumerate(solutions):
    s = np.array(s)
    x0s = next(nearest_val(s[np.arange(0, len(s), 1) % 3 == 2], 1367))
    plt.polar(np.pi*nr/18, s[x0s*3], '.', color=[1,1,1], ms=0.7)


# %%
