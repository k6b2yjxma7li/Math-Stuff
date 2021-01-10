# %%
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection as mpc
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
# api for data extraction
import data_import as di
# api for convolutions and fft
import spectrals as sp

# global settings for datafiles
di.PATH = ".data/Praca_inzynierska/Badania/200924/polar_si/VV"
di.XNAME = "#Wave"
di.YNAME = "#Intensity"

# read all datafiles
di.read_dir()
# initialize globals X and Y
di.get_data()

# global setting for plot style
plt.style.use('classic')


def polygon(a, b, botlim=0):
    """Generates vertices for polygon to fill space under plot line"""
    return [(a[0], botlim), *zip(a, b), (a[-1], botlim)]


# data lines averages
x_av, y_av = np.mean(list(di.DATA.values()), 0).T
# data lines std devs
x_sd, y_sd = np.std(list(di.DATA.values()), 0).T

# %%
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(111, projection='3d')
# ranges
y_min = max(di.Y)
y_max = 0

fillup = []
# plots and range finding
for fnr in range(37):
    x, y = di.get_data(file_no=fnr)
    # cutting data
    fltr = 200 < x
    # zdir='y' sets vertical axis as z axis (x-y plane as ground)
    ax.plot(*[x[fltr], y[fltr], fnr*10], '-', lw=1, color='black', zdir='y')
    # get lowest y point for all y-sets
    if y_min > min(y[fltr]):
        y_min = min(y[fltr])
    if y_max < max(y[fltr]):
        y_max = max(y[fltr])

# average plot - peaks and more
fltr = x_av > 200
ax.plot(x_av[fltr], y_sd[fltr], zs=370, zdir='y')

# settings for all axes
label_kwargs = dict(usetex=True, fontsize=15, fontweight='bold')
# axes labels
ax.set_xlabel(r'\(\Delta \nu [\mathrm{cm^{-1}}]\)', **label_kwargs)
ax.set_ylabel(r'Polaryzacja \([^{\circ}]\)', **label_kwargs)
# labelpad=-10 pulls axis label closer to axis spine
ax.set_zlabel(r'Intensywność [j.u.]', labelpad=-10, **label_kwargs)

# axes limits
ax.set_xlim([min(di.X[fltr]), max(di.X)])
ax.set_ylim([0, 360])
# 'advanced' scaling for z axis limits
min_add_scale = 0.1
max_add_scale = min_add_scale
ax.set_zlim([y_min-min_add_scale*abs(y_min),
             y_max+max_add_scale*abs(y_max)])

# under-lines filling
for fnr in range(37):
    x, y = di.get_data(file_no=fnr)
    fltr = 200 < x
    fillup.append(polygon(x[fltr], y[fltr], y_min-min_add_scale*abs(y_min)))
ax.add_collection3d(mpc(fillup, facecolors=["white"]*37),
                    np.arange(0, 370, 10), zdir='y')

# removing z ticks (arbitrary units)
ax.set_zticks([])

# x and y axes tick locators
# ax.xaxis.set_minor_locator(MultipleLocator(20))
ax.xaxis.set_major_locator(MultipleLocator(100))

ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(30))

# does not work (idk why)
ax.xaxis.set_tick_params(which='major', length=10)
ax.xaxis.set_tick_params(which='minor', length=2)

# clearing 3d box
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# camera settings
ax.elev = 20
ax.azim = -60
ax.dist = 10

# 3d box grid settings
ax.grid(False)
# does not work (idk why)
ax.zaxis.grid(True)

# same as tight layout, more adjustable
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

fig.show()
# %%
fig.savefig("all_3d.png", dpi=300)

# %%
