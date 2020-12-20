# %%
import numpy as np
import plotly.express as pex
import plotly.subplots as psp
import plotly.graph_objects as pgo


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


# # %%
# Incident light wave vector
k = np.array([0.5, -0.1, -1])

phi, theta, psi = k_vec(*k)

R_i = (R_phi(phi).dot(R_theta(theta)).dot(R_psi(psi)))
R = R_phi(phi).dot(R_theta(theta)).dot(R_psi(psi))

a, b, d = 1, 1, 1
psi_tensor = 0
# rotating tensor around Z axis


A1g = np.array([[a, 0, 0],
                [0, a, 0],
                [0, 0, b]])

E2g1 = np.array([[0, -d, 0],
                 [-d, 0, 0],
                 [0, 0, 0]])

E2g2 = np.array([[d, 0, 0],
                 [0, -d, 0],
                 [0, 0, 0]])

mode = {
    'A1g': A1g,
    'E2g1': E2g1,
    'E2g2': E2g2
}

p_type = 'VH'


# angles
t = np.arange(0, 2*np.pi*(1 + 1/36), 2*np.pi/36)
txt = [f"{10*n} deg." for n in range(len(t))]

# direction vectors
uni = np.array([np.cos(t), np.sin(t), np.zeros(len(t))])
# polarizations
Eik = uni.copy() * k.dot(k)**0.5
# incident field vectors
Ei = R_i.dot(Eik)
# measurement vectors
uni_m = R_i.dot(uni)

Es = np.array([np.zeros(len(t)),
               np.zeros(len(t)),
               np.zeros(len(t))])

# response vectors, R_psi is a mode tensor rotation around z-axis
modes = ['E2g1', 'E2g2']
for md in modes:
    Es += R_psi(psi_tensor).dot(mode[md]).dot(Ei)

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
    {
        'x': [0, k[0]],
        'y': [0, k[1]],
        'z': [0, k[2]],
        'type': 'scatter3d',
        'mode': 'lines',
        'line': {
            'color': 'white',
            'width': 2
        },
        'name': 'k'
    },
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

layout = {
    'title': f'{modes} seen from {k} ({p_type})',
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

fig.add_traces(basis + endpoints + traces)
fig.update_layout(layout)

fig.show()

# %%
