# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('default')
polar_conf = 'VV'
# mpl.rcParams['text.usetex'] = True


def R_psi(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])


def tensor_plot(theta: np.array, v_inc: np.array, tensor_fun, name=None,
                config='VV'):

    if '__len__' not in dir(tensor_fun):
        tensor_fun = [tensor_fun]
    if name is None:
        name = r'\(\)'
    fig = plt.figure(figsize=[3*len(tensor_fun), 3])
    ax = []
    if type(name) in [list, tuple]:
        if len(name) > 1 or len(name) == 0:
            name = r''.join([r'\(\mathrm{' + nm + r'}\), ' for nm in name])
        else:
            name = r'\(\mathrm{' + name[0] + r'}\)'
    else:
        name = r'\(\mathrm{'+name+r'}\)'
    if len(tensor_fun) > 1:
        fig.suptitle(r'Tensory '+name, usetex=True)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.8)
    else:
        fig.suptitle(r'Tensor '+name, usetex=True)
        plt.subplots_adjust(top=0.8)
    id = 100 + 10*len(tensor_fun)
    R = R_psi(-np.pi/2)
    phi = -np.pi/6
    Rp = R_psi(phi)
    Rpi = la.inv(R_psi(phi))
    for nr, tensor in enumerate(tensor_fun):
        if config == 'VV':
            intensity = (v_inc.dot(Rp.dot(tensor_fun[nr](1)).dot(Rpi).
                         dot(v_inc))**2)
        elif config == 'VH':
            v_vh = R.dot(v_inc)
            intensity = (v_vh.dot(Rp.dot(tensor_fun[nr](1)).dot(Rpi).
                                  dot(v_inc))**2)
        ax.append(plt.subplot(id + nr + 1, projection='polar'))
        ax[nr].set_yticklabels([])
        ax[nr].plot(theta, intensity, lw=2)
        # requirement for 64bit numpy floats
        if max(intensity) > 1e-16:
            ax[nr].set_ylim(min(intensity)*0.001, max(intensity)*1.1)
        else:
            ax[nr].set_ylim(-0.001, 1.1)
    fig.show()


## %%
n_steps = 360
t = np.arange(0, 2*np.pi*(1+1/n_steps), 2*np.pi/n_steps)
ei = np.array([np.cos(t), np.sin(t), 0])

A_1 = lambda a: np.array([[a, 0, 0],
                          [0, a, 0],
                          [0, 0, a]])

E1 = lambda a: np.array([[a, 0, 0],
                         [0, a, 0],
                         [0, 0, -2*a]])

E2 = lambda a: np.array([[-3**0.5 * a, 0, 0],
                         [0, 3**0.5 * a, 0],
                         [0, 0, 0]])

T_2x = lambda a: np.array([[0, 0, 0],
                           [0, 0, a],
                           [0, a, 0]])

T_2y = lambda a: np.array([[0, 0, a],
                           [0, 0, 0],
                           [a, 0, 0]])

T_2z = lambda a: np.array([[0, a, 0],
                           [a, 0, 0],
                           [0, 0, 0]])
## %%
ftype = '.png'
dpi = 100

tensor_plot(t, ei, A_1, 'A_1', config=polar_conf)
plt.savefig('a1_mode'+ftype, dpi=dpi)

tensor_plot(t, ei, [E1, E2], 'E', config=polar_conf)
plt.savefig('e_mode'+ftype, dpi=dpi)

tensor_plot(t, ei, T_2x, 'T_2(x)', config=polar_conf)
plt.savefig('t2x_mode'+ftype, dpi=dpi)
tensor_plot(t, ei, T_2y, 'T_2(y)', config=polar_conf)
plt.savefig('t2y_mode'+ftype, dpi=dpi)
tensor_plot(t, ei, T_2z, 'T_2(z)', config=polar_conf)
plt.savefig('t2z_mode'+ftype, dpi=dpi)

# %%
