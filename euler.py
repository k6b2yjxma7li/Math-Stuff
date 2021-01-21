# %%
from scipy.optimize import leastsq

import plotly.graph_objects as pgo
import plotly.subplots as psp
import plotly.express as pex
import pandas as pd
import numpy as np
import numpy.linalg as la
import os


def vp_vec(phip):
    return np.array([np.cos(phip), np.sin(phip), 0])


def k_vec(x, y, z, *args):
    """
    phi, theta, psi
    """
    norm = (x**2 + y**2 + z**2)**0.5
    xn = x/norm
    yn = y/norm
    zn = z/norm
    return (np.arctan2(zn, yn) - np.pi/2,
            np.arctan2(xn, (yn**2 + zn**2)**0.5), 0)


def R_phi(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])


def R_theta(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def R_psi(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])


def response(data, tensors, polarization='VV'):
    """Tensor list `tensors` is required to be nested array-like object.
    (1) First layer defines number of tensor groups - each tensor in a group
    share parameters.

    (2) Second layer defines number of tensors in a group - intensity is
    calculated for each tensor separately, response from phonon is then
    calculated as a sum of intensities from tensors in a single group.

    (3) Third layer defines number of parameters in each tensor - some tensors
    have more than one uncorrelated parameters and all of them have to get it's
    place; subtensors from this layer are multiplied by corresponding parameter
    to define full tensor."""

    # angles
    t = np.arange(0, 2*np.pi*(1 + 1/36), 2*np.pi/36)
    # polarization directions
    uni = np.array([np.cos(t), np.sin(t), np.zeros(len(t))])
    # measurement direction
    if polarization == 'VV':
        uni_m = uni.copy()
    elif polarization == 'VH':
        rot90 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        uni_m = rot90.dot(uni)

    def _resp_(parameters):
        # rotation matrices angles
        # phi, theta, psi = parameters[:3]
        phi, theta, psi = k_vec(*[0, 0, -1])
        # crystal lattice z rotation angle
        psi0 = parameters[3]
        # laser wave number
        k = parameters[4]
        # tensor parameters
        t_params = parameters[5:]

        # all rotations matrix
        R = R_phi(phi).dot(R_theta(theta)).dot(R_psi(psi))
        # incident electrical polarizations
        Ei = R.dot(uni.copy() * k)
        # scattered light electrical polarizations
        Es = np.array([np.zeros(len(t)),
                       np.zeros(len(t)),
                       np.zeros(len(t))])
        # creating proper tensors
        t_param_ix = 0
        tenors_evaluated = []
        # tensor groups (1)
        for t_group in tensors:
            # tensors (2)
            for tensor in t_group:
                # zero tensor added to list
                tenors_evaluated.append(np.array([
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ], dtype=np.float64))
                # tensor components (3)
                for tnr, t_comp in enumerate(tensor):
                    # evaluation of parameter into tensor component
                    # then adding component to the real tensor
                    tenors_evaluated[-1] += t_comp*t_params[tnr+t_param_ix]
                    # tracking tensor components
            t_param_ix += tnr+1
        # for each real tensor response is calculated
        # and added to overall response
        Rt = R_psi(psi0)
        Rti = la.inv(Rt)
        for tensor_value in tenors_evaluated:
            Es += Rt.dot(tensor_value).dot(Rti).dot(Ei)
        # measurement values of intensity
        Em = np.sum(R.dot(uni_m) * Es, 0)**2
        return data - Em
    return _resp_


p_type = 'VH'
material = 'si'

path = f"./rad_{material}.csv"

if os.path.isfile(path):
    data = pd.read_csv(path, engine='python')
data_result = np.zeros(len(data))
for col in data:
    if p_type in str(col):
        print(col)
        data_result += np.array(data[col])

tensors = [[[np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])],
            [np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])],
            [np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])]]]

# starting = list(k_vec(*[0, 0, -1]))+[0, 1000]+[1]
# par, h = leastsq(response(data_result, tensors, p_type), starting)

# # %%
# Incident light wave vector
k = np.array([0.2, -0.2, -1])

phi, theta, psi = k_vec(*k)
# phi, theta, psi, psi0, k = par[:5]

# d, = par[5:]
d = 1
psi0 = np.pi/2

R = R_phi(phi).dot(R_theta(theta)).dot(R_psi(psi))
Ri = R_psi(psi).T.dot(R_theta(theta).T).dot(R_phi(phi).T)

a, b = 1, 1
# rotating tensor around Z axis
psi_tensor = 0

A1 = np.array([
    [a, 0, 0],
    [0, a, 0],
    [0, 0, a]
])

A1g = np.array([[a, 0, 0],
                [0, a, 0],
                [0, 0, b]])

E1 = np.array([
    [b, 0, 0],
    [0, b, 0],
    [0, 0, -2*b]
])

E2 = np.array([
    [-3**0.5 * b, 0, 0],
    [0, 3**0.5 * b, 0],
    [0, 0, 0]
])

E2g1 = np.array([[0, -d, 0],
                 [-d, 0, 0],
                 [0, 0, 0]])

E2g2 = np.array([[d, 0, 0],
                 [0, -d, 0],
                 [0, 0, 0]])

T2x = np.array([
    [0, 0, 0],
    [0, 0, d],
    [0, d, 0]
])

T2y = np.array([
    [0, 0, d],
    [0, 0, 0],
    [d, 0, 0]
])

T2z = np.array([
    [0, d, 0],
    [d, 0, 0],
    [0, 0, 0]
])

mode = {
    'A1': A1,
    'A1g': A1g,
    'E1': E1,
    'E2': E2,
    'E2g1': E2g1,
    'E2g2': E2g2,
    'T2x': T2x,
    'T2y': T2y,
    'T2z': T2z,
}

# # %%
# angles
t = np.arange(0, 2*np.pi*(1 + 1/36), 2*np.pi/36)
txt = [f"{10*n} deg." for n in range(len(t))]

# direction vectors
uni = np.array([np.cos(t), np.sin(t), np.zeros(len(t))])
# polarizations
if '__len__' in dir(k):
    if len(k) == 3:
        Eik = uni.copy() * k.dot(k)**0.5
else:
    Eik = uni.copy() * k
# incident field vectors
Ei = R.dot(Eik)
# measurement vectors
uni_m = R.dot(uni)

Es = np.array([np.zeros(len(t)),
               np.zeros(len(t)),
               np.zeros(len(t))])

# response vectors, R_psi is a mode tensor rotation around z-axis
modes = ['E2g1', 'E2g2']
for md in modes:
    Es += R_psi(psi0).dot(mode[md]).dot(la.inv(R_psi(psi0))).dot(Ei)

# measurement values
if p_type == 'VV':
    Em = uni_m * np.sum(uni_m * Es, 0)**2
elif p_type == 'VH':
    Em = uni_m * np.sum(R_psi(np.pi/2).dot(uni_m) * Es, 0)**2

# unitary vectors
i_v = [1, 0, 0]
j_v = [0, 1, 0]
k_v = [0, 0, 1]

i_v = R.dot(i_v)
j_v = R.dot(j_v)
k_v = R.dot(k_v)
# # %%
fig = pex.scatter_3d()
fig.layout.template = 'plotly_dark'

endpoints = [
    # endpoints
    # X'
    {
        'x': [i_v[0]],
        'y': [i_v[1]],
        'z': [i_v[2]],
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'size': 5,
            'color': 'red'
        },
        'name': 'i (X\')'
    },
    # Y'
    {
        'x': [j_v[0]],
        'y': [j_v[1]],
        'z': [j_v[2]],
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'size': 5,
            'color': 'green'
        },
        'name': 'j (Y\')'
    },
    # Z'
    {
        'x': [k_v[0]],
        'y': [k_v[1]],
        'z': [k_v[2]],
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'size': 5,
            'color': 'blue'
        },
        'name': 'k (Z\')'
    }
]

basis = [
    # vecs
    # X'
    {
        'x': [0, i_v[0]],
        'y': [0, i_v[1]],
        'z': [0, i_v[2]],
        'type': 'scatter3d',
        'mode': 'lines',
        'line': {
            'width': 2,
            'color': 'red'
        },
        'name': 'i (X\')'
    },
    # Y'
    {
        'x': [0, j_v[0]],
        'y': [0, j_v[1]],
        'z': [0, j_v[2]],
        'type': 'scatter3d',
        'mode': 'lines',
        'line': {
            'width': 2,
            'color': 'green'
        },
        'name': 'j (Y\')'
    },
    # Z'
    {
        'x': [0, k_v[0]],
        'y': [0, k_v[1]],
        'z': [0, k_v[2]],
        'type': 'scatter3d',
        'mode': 'lines',
        'line': {
            'width': 2,
            'color': 'blue'
        },
        'name': 'k (Z\')'
    },
    # normal
    # X
    {
        'x': [0, 1],
        'y': [0, 0],
        'z': [0, 0],
        'type': 'scatter3d',
        'mode': 'lines',
        'line': {
            'width': 5,
            'color': 'red'
        },
        'name': 'i (X)'
    },
    # Y
    {
        'x': [0, 0],
        'y': [0, 1],
        'z': [0, 0],
        'type': 'scatter3d',
        'mode': 'lines',
        'line': {
            'width': 5,
            'color': 'green'
        },
        'name': 'j (Y)'
    },
    # Z
    {
        'x': [0, 0],
        'y': [0, 0],
        'z': [0, 1],
        'type': 'scatter3d',
        'mode': 'lines',
        'line': {
            'width': 5,
            'color': 'blue'
        },
        'name': 'k (Z)'
    },
    # O point
    {
        'x': [0],
        'y': [0],
        'z': [0],
        'type': 'scatter3d',
        'marker': {
            'size': 5,
            'color': 'white'
        },
        'name': 'O (XYZ)'
    }
]

traces = [
    # k vector
    # {
    #     'x': [0, k[0]],
    #     'y': [0, k[1]],
    #     'z': [0, k[2]],
    #     'type': 'scatter3d',
    #     'mode': 'lines',
    #     'line': {
    #         'color': 'white',
    #         'width': 2
    #     },
    #     'name': 'k'
    # },
    # E incident field vecs
    {
        'x': Ei[0],
        'y': Ei[1],
        'z': Ei[2],
        'text': txt,
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'color': 'yellow',
            'size': 2
        },
        'name': 'incident field'
    },
    # E scattered field vecs
    {
        'x': Es[0],
        'y': Es[1],
        'z': Es[2],
        'text': txt,
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'color': 'magenta',
            'size': 1
        },
        'name': 'scattered field'
    },
    # E measured field vecs
    {
        'x': Em[0],
        'y': Em[1],
        'z': Em[2],
        'text': txt,
        'type': 'scatter3d',
        'mode': 'lines',
        'line': {
            'color': 'magenta',
            # 'size': 3
            'width': 2
        },
        'name': 'measured field'
    }
]

real_response = R.dot(uni)*data_result
data_traces = [
    {
        # 'x': real_response[0],
        # 'y': real_response[1],
        # 'z': real_response[2],
        # 'text': txt,
        # 'type': 'scatter3d',
        # 'mode': 'lines',
        # 'line': {
        #     'color': 'white',
        #     'width': 2
        # },
        # 'name': 'Data'
    }
]

title_part = str(modes).replace('[', '')
title_part = title_part.replace(']', '')
title_part = title_part.replace('\'', '')

layout = {
    'title': f'{title_part} seen from {k} ({p_type})',
    'title_x': 0.5,
    'title_y': 0.95,
    'scene': {
        'xaxis': {
            'title': 'x'
        },
        'yaxis': {
            'title': 'y'
        },
        'zaxis': {
            'title': 'z'
        }
    },
    'margin': {
        't': 25,
        'b': 5,
        'l': 5,
        'r': 5
    }
}

fig.add_traces(basis + endpoints + traces + data_traces)
fig.update_layout(layout)

fig.show()

# %%
import matplotlib.pyplot as plt
resp = response(np.linspace(0, 0, len(t)), [[[E2g1], [E2g2]]])

plt.polar(t, -resp(list(k_vec(*[0, 0, -1]))+[0, 1, 1]))

# %%
