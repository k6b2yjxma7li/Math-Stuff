# %%
import numpy as np
import plotly.express as pex
import plotly.subplots as psp
import plotly.graph_objects as pgo
# %%


def vp_vec(phip):
    return np.array([np.cos(phip), np.sin(phip), 0])


def k_vec(x, y, z, *args):
    """
    phi, theta
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
k = np.array([0, 0, -1])

phi, theta, psi = k_vec(*k)

R_i = (R_phi(phi).dot(R_theta(theta)).dot(R_psi(psi)))
R = R_phi(phi).dot(R_theta(theta)).dot(R_psi(psi))

E_x = []
E_y = []
E_z = []

a, b, d = 1, 0.1, 1
psi_tensor = 0
tensor_rotz = np.array([[np.cos(psi_tensor), -np.sin(psi_tensor), 0],
                        [np.sin(psi_tensor), np.cos(psi_tensor), 0],
                        [0, 0, 1]])

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

curr_mode = 'E2g2'
p_type = 'VV'

Es_x = []
Es_y = []
Es_z = []

Em_x = []
Em_y = []
Em_z = []

Es_xp = []
Es_yp = []
Es_zp = []

nfold = 36

txt = []

for n in range(nfold):
    # dla tesnora E2g należy wziąć sumę intensywności z obu tensorów
    txt.append(f"{10*n} deg.")
    E = np.array([np.cos(2*np.pi/nfold * n),
                  np.sin(2*np.pi/nfold * n),
                  0])
    E = R_i.dot(E)*(k.dot(k))**0.5 # incident polarization vector
    if mode == 'E2g1':
        Es1 = tensor_rotz.dot(mode['E2g2']).dot(E) # response polarization vector
        Em += abs(uni.dot(tensor_rotz.dot(mode['E2g2']).dot(E)))**2
    if mode == 'E2g2':
        Es1 = tensor_rotz.dot(mode['E2g1']).dot(E) # response polarization vector
        Em += abs(uni.dot(tensor_rotz.dot(mode['E2g1']).dot(E)))**2
    Es = tensor_rotz.dot(mode[curr_mode]).dot(E) # response polarization vector
    if p_type == 'VV':
        uni = R_i.dot(np.array([np.cos(2*np.pi/nfold * n),
                                np.sin(2*np.pi/nfold * n),
                                0]))
        Em = abs(uni.dot(Es))**2 * E/(E.dot(E))
    if p_type == 'VH':
        uni = R_i.dot(np.array([-np.sin(2*np.pi/nfold * n),
                                np.cos(2*np.pi/nfold * n),
                                0]))
        Em = abs(uni.dot(Es))**2 * E/(E.dot(E))
    
    if curr_mode == 'E2g1':
        Em += abs(uni.dot(tensor_rotz.dot(mode['E2g2']).dot(E)))**2
    if curr_mode == 'E2g2':
        Em += abs(uni.dot(tensor_rotz.dot(mode['E2g1']).dot(E)))**2

    if curr_mode in ['E2g1', 'E2g2']:
        Es_xp.append(Es[0])
        Es_yp.append(Es[1])
        Es_zp.append(Es[2])

    Es_x.append(Es[0])
    Es_y.append(Es[1])
    Es_z.append(Es[2])

    Em_x.append(Em[0])
    Em_y.append(Em[1])
    Em_z.append(Em[2])

    E_x.append(E[0])
    E_y.append(E[1])
    E_z.append(E[2])

i_v = [1, 0, 0]
j_v = [0, 1, 0]
k_v = [0, 0, 1]

i_v = R.dot(i_v)
j_v = R.dot(j_v)
k_v = R.dot(k_v)

fig = pex.scatter_3d()
fig.layout.template = 'plotly_dark'

traces = [
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
    },
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
    },
    # v vector
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
        'name': 'v'
    },
    # E field vecs
    {
        'x': E_x,
        'y': E_y,
        'z': E_z,
        'text': txt,
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'color': 'yellow',
            'size': 2
        },
        'name': 'incident field'
    },
    # Es field vecs
    {
        'x': Es_x,
        'y': Es_y,
        'z': Es_z,
        'text': txt,
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'color': 'magenta',
            'size': 1
        },
        'name': 'scattered field'
    },
    # Em field vecs
    {
        'x': Em_x,
        'y': Em_y,
        'z': Em_z,
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
    'title': f'Mode {curr_mode} seen from {k} ({p_type})',
    'title_x': 0.5,
    'title_y': 0.95,
    'scene': {
        'xaxis': {
            'title': 'x'
        },
        'yaxis': {
            'title': 'y'
            # 'scaleanchor': 'x',
            # 'scaleratio': 1
        },
        'zaxis': {
            'title': 'z'
            # 'scaleanchor': 'x',
            # 'scaleratio': 1
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