# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('default')
# mpl.rcParams['text.usetex'] = True


def tensor_plot(theta: np.array, v_inc: np.array, tensor_fun, name=None):

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
    for nr, tensor in enumerate(tensor_fun):
        intensity = v_inc.dot(tensor_fun[nr](1).dot(v_inc))**2
        ax.append(plt.subplot(id + nr + 1, projection='polar'))
        ax[nr].set_yticklabels([])
        ax[nr].plot(theta, intensity, lw=2)
        if max(intensity) > 0:
            ax[nr].set_ylim(min(intensity)*0.001, max(intensity)*1.1)
        else:
            ax[nr].set_ylim(-0.001, 1.1)
    fig.show()


## %%
n_steps = 100
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

tensor_plot(t, ei, A_1, 'A_1')
plt.savefig('a1_mode'+ftype, dpi=dpi)

tensor_plot(t, ei, [E1, E2], 'E')
plt.savefig('e_mode'+ftype, dpi=dpi)

tensor_plot(t, ei, T_2x, 'T_2(x)')
plt.savefig('t2x_mode'+ftype, dpi=dpi)
tensor_plot(t, ei, T_2y, 'T_2(y)')
plt.savefig('t2y_mode'+ftype, dpi=dpi)
tensor_plot(t, ei, T_2z, 'T_2(z)')
plt.savefig('t2z_mode'+ftype, dpi=dpi)

# %%
