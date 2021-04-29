# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection as mpc
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator,
                               FixedLocator)
import matplotlib.gridspec as mgs
# api for data extraction
import nano.data_import as di
# api for convolutions and fft (top layer of numpy fft)
import nano.spectrals as sp
from nano.spectrals import d, kernel, spectrum, residual
import json as js
# fitting
from scipy.optimize import leastsq

# checked equations and functions based on them
import symping as sy
from arc import arc_2D


# global settings for datafiles
dset = "VH"
tag = "grf"

# global setting for plot style
plt.style.use('default')

bands = []

modes_tagged = {
    "grf": ["2D", "G", "D"],
    "hbn": [r"\)mod \(E_{2g}"]
}

colors = ['#ff6600', '#0066ff', '#33ff33']

mode_names = modes_tagged[tag]

config_files = {
    "VV": f"{tag}_VV.json",
    "VH": f"{tag}_VH.json"
}

config = {}

with open(config_files[dset], 'r') as confstream:
    _conf_ = js.load(confstream)
    config = _conf_['materials'][tag]
    bands = config['bands']


di.PATH = config['path']
di.XNAME = "#Wave"
di.YNAME = "#Intensity"


di.FILES = []
# read all datafiles
di.read_dir()
# initialize globals X and Y
di.get_data()


def polygon(a, b, botlim=0):
    """Generates vertices for polygon to fill space under plot line"""
    return [(a[0], botlim), *zip(a, b), (a[-1], botlim)]


# data lines averages
x_av, y_av = np.mean(list(di.DATA.values()), 0).T
# data lines stddevs
x_sd, y_sd = np.std(list(di.DATA.values()), 0).T


# %%
# PART 1 -- showing all datasets
# Proper plotting
fig = plt.figure(figsize=(12, 8))
plot_gs = mgs.GridSpec(nrows=3, ncols=2, figure=fig)
gs_plot_a = plot_gs[0:2, 0:1]
gs_plot_b = plot_gs[2:, 0:1]
# Section 1
ax = fig.add_subplot(gs_plot_a, projection='3d')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.1, hspace=0.1)
# ranges
y_min = max(di.Y)
y_max = 0

# range finding
for fnr in range(37):
    x, y = di.get_data(file_no=fnr)
    # cutting data
    fltr = 200 < x
    if y_min > min(y[fltr]):
        y_min = min(y[fltr])
    if y_max < max(y[fltr]):
        y_max = max(y[fltr])

# settings for all axes
label_kwargs = dict(usetex=True, fontsize=12, fontweight='bold')
# axes labels
ax.set_xlabel(r'\(\Delta \nu [\mathrm{cm^{-1}}]\)', labelpad=10,
              **label_kwargs)
ax.set_ylabel(r'Polaryzacja \([^{\circ}]\)', labelpad=5, **label_kwargs)
# labelpad=-10 pulls axis label closer to axis spine
ax.set_zlabel(r'Amplituda [j.u.]', labelpad=-10, **label_kwargs)


# axes limits
ax.set_xlim([min(di.X[fltr]), max(di.X)])
ax.set_ylim([0, 360])
# 'advanced' scaling for z axis limits
min_add_scale = 0.1
max_add_scale = min_add_scale
zlim_bot = y_min-min_add_scale*abs(y_min)
zlim_top = y_max+max_add_scale*abs(y_max)

ax.set_zlim([zlim_bot, zlim_top])

# Plotting:
# under-lines filling
fillup = []
for fnr in range(37):
    x, y = di.get_data(file_no=fnr)
    fltr = 200 < x
    fillup.append(polygon(x[fltr], y[fltr], y_min-min_add_scale*abs(y_min)))
# zdir='y' sets vertical axis as z axis (x-y plane as ground)
ax.add_collection3d(mpc(fillup, facecolors=["white"]*37, lw=0.4,
                        edgecolor='black'), np.arange(0, 370, 10), zdir='y')

# peak position line
x_av_ranged = x_av[fltr]
y_av_ranged = y_av[fltr]

# 2 lines (L shape)
for mode_name, center in zip(mode_names, bands['center']):
    if dset == 'VV' and tag == 'grf':
        zlim_top += 4000
    else:
        zlim_top += 2000
    ax.plot(*[[center, center], [zlim_bot, zlim_top], 370],
            '--', lw=2, zdir='y', color='black')
    ax.plot(*[[center, center], [y_min, y_min], [0, 370]],
            '--', lw=2, zdir='y', color='black')

    # line label
    label = (r'\('+mode_name+r'\): \('+str(round(center, 1)) +
             r'\mathrm{cm^{-1}}\)')
    # zdir does not interchange z axis, have to use true coords
    # (y and z swapped)
    ax.text(center, 370, zlim_top, label, fontsize=15, color='black',
            usetex=True, zdir=(1, 0, 0))


# Plot parameters:
# removing z ticks (arbitrary units)
ax.set_zticks([])

# x and y axes tick locators
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(30))

# clearing 3d box
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# adjusting tick labels rotation
if tag == 'grf':
    ax.set_xticklabels(np.array(ax.xaxis.get_ticklocs(), dtype=int),
                       ha='left',
                       # va='center',
                       fontsize=10)
    ax.xaxis.set_tick_params(rotation=-40, pad=-8)
    ax.set_yticklabels(np.array(ax.yaxis.get_ticklocs(), dtype=int),
                       ha='right',
                       # va='center',
                       fontsize=10)
    ax.yaxis.set_tick_params(rotation=20, pad=-7)
if tag == 'hbn':
    ax.set_xticklabels(np.array(ax.xaxis.get_ticklocs(), dtype=int),
                       ha='right',
                       # va='center',
                       fontsize=10)
    ax.xaxis.set_tick_params(rotation=40, pad=-8)
    ax.set_yticklabels(np.array(ax.yaxis.get_ticklocs(), dtype=int),
                       ha='left',
                       # va='center',
                       fontsize=10)
    ax.yaxis.set_tick_params(rotation=-20, pad=-7)

# does not work (idk why)
ax.xaxis.set_tick_params(which='major', length=10)
ax.xaxis.set_tick_params(which='minor', length=2)


# camera settings
if tag == 'grf':
    ax.elev = 20
    ax.azim = -110
    ax.dist = 10
if tag == 'hbn':
    ax.elev = 20
    ax.azim = -60
    ax.dist = 10

# 3d box grid settings
ax.grid(False)
# does not work (idk why)
# ax.zaxis.grid(True)

# same as tight layout, more adjustable
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


# Section 2
# new subplot, new ax
ax2 = fig.add_subplot(gs_plot_b, title=r"Intensywności ("+dset+")")
for mode, color, mode_name in zip(bands['intensity'], colors, mode_names):
    # plot
    if tag == 'grf':
        polar_data = mode
        ax2.plot(range(0, 370, 10), polar_data.copy(), '.', ms=10, color=color,
                 label=mode_name)
    else:
        polar_data = mode
        ax2.plot(range(0, 370, 10), polar_data.copy(), '.', ms=10,
                 color='black', label=mode_name)
ax2.set_xlabel(r'Polaryzacja \([^{\circ}]\)', **label_kwargs)
ax2.set_ylabel(r'Intensywność [j.u]', **label_kwargs)
if len(bands['intensity']) > 1:
    ax2.legend()

# ticks
# ax2.set_yticks([])
ax2.yaxis.set_major_locator(FixedLocator([0]))
ax2.xaxis.set_major_locator(MultipleLocator(30))
ax2.xaxis.set_minor_locator(MultipleLocator(10))

# axes limits
if tag == 'hbn':
    ax2.set_ylim(0, 1.2*max(polar_data))
ax2.set_xlim([-5, 365])

# forcing shapes


# deg grid
ax2.grid(True, which='major', lw=0.5, ls='--', color='black')
ax2.grid(True, which='minor', lw=0.5, ls=':', color='black')

# finish
fig.show()
fig.savefig(f"{tag}_all_3d_{dset}.png", dpi=300, bbox_inches='tight')


# %%
# PART 3 -- fitting to intensity
if tag == 'grf':
    polar_data = (np.array(bands['intensity'][0]) /
                  np.array(bands['intensity'][1]))
else:
    polar_data = np.array(bands['intensity'][0])
# polar_data = np.array(bands['intensity'][0])

fig = plt.figure(figsize=[6, 6])
ax = fig.add_subplot(111, projection='polar')
ax.set_title(f"Składowa {dset}")
dphi = np.pi/18
phi = np.arange(0, 2*np.pi+dphi, dphi)

if dset == 'VV':
    def fit_fun(x, p):
        if len(p) > 2:
            theta = p[2]
        else:
            theta = np.pi/2
        return sy.IT2_np(sy.Em_VV, p[0], x-p[1] % np.pi, theta)
if dset == 'VH':
    def fit_fun(x, p):
        if len(p) > 2:
            theta = p[2]
        else:
            theta = np.pi/2
        return sy.IT2_np(sy.Em_VH, p[0], x-p[1] % np.pi, theta)


def res(p):
    return polar_data - fit_fun(phi, p)


markers = ['*-', '^-', '.-']
if tag in ['hbn', 'si']:
    ax.plot(phi, polar_data, '.-', ms=10, color='black', label="Intensywność")
else:
    for band, clr, name, mrkr in zip(bands['intensity'], colors,
                                     mode_names, markers):
        ax.plot(phi, np.array(band)/max(band), mrkr, ms=10, color=clr,
                label=name)
ax.set_yticklabels([])
# ax.plot(phi, polar_data, '.', ms=10, color='black', label="2D/G")
p = np.array([0, 0, 0])
# p, h = leastsq(res, [1000, 0.1])

d_th = np.pi/180

theta = np.arange(0, 2*np.pi+d_th, d_th)

# ax.plot(theta, fit_fun(theta, p), color='black', lw=0.7, label="Dopasowanie")
# ax.quiver(*[0, 0], *[0, max(polar_data)*1.1], scale=1, scale_units='xy',
#           angles='xy', color='grey')
ax.set_ylim(0, None)
ax.legend()
fig.show()
phi_min = abs((p[1] % np.pi)) - np.pi
# ax.quiver(*[0, 0], *[phi_min, max(polar_data)*1.1], scale=1,
#           scale_units='xy', angles='xy', color='black')

# arc_2D(ax, [0, 0], [3*max(polar_data)/4, 0], phi_min, text=r'$\phi_0$',
#        projection='polar', color='black', lw=0.7, ls='--')

print(f"Phi0 {180*phi_min/np.pi} deg.")
if len(p) > 2:
    theta_min = abs(180*(p[2] % np.pi)/np.pi)
    print(f"Theta {theta_min} deg.")
if tag == 'grf':
    fig.savefig(f'{tag}_{dset}_bands.png', dpi=300, bbox_inches='tight')
else:
    fig.savefig(f'{tag}_polar_w_fit_{dset}.png', dpi=300, bbox_inches='tight')

# %%
