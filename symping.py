# %%
import sympy as sm
import sympy.matrices as smm
from sympy.interactive.printing import init_printing
import mpmath
import sys

import numpy as np
import matplotlib.pyplot as plt

init_printing(use_unicode=True, wrap_line=False)
sys.modules['sympy.mpmath'] = mpmath

component = 'VH'

a, b, c, d, p, t = sm.symbols(r'a b c d \phi \theta')

# symbol matrices
# si
A_1 = smm.Matrix([[a, 0, 0],
                  [0, a, 0],
                  [0, 0, a]])

E1 = smm.Matrix([[b, 0, 0],
                 [0, b, 0],
                 [0, 0, -2*b]])

E2 = smm.Matrix([[-sm.sqrt(3) * b, 0, 0],
                 [0, sm.sqrt(3) * b, 0],
                 [0, 0, 0]])

T_2x = smm.Matrix([[0, 0, 0],
                   [0, 0, d],
                   [0, d, 0]])

T_2y = smm.Matrix([[0, 0, d],
                   [0, 0, 0],
                   [d, 0, 0]])

T_2z = smm.Matrix([[0, d, 0],
                   [d, 0, 0],
                   [0, 0, 0]])

# grf/hbn
A1g = smm.Matrix([[a, 0, 0],
                  [0, a, 0],
                  [0, 0, b]])

E2g1 = smm.Matrix([[0, -d, 0],
                   [-d, 0, 0],
                   [0, 0, 0]])

E2g2 = smm.Matrix([[d, 0, 0],
                   [0, -d, 0],
                   [0, 0, 0]])


# functions of symbol matrices
def R_x(alpha): return smm.Matrix([[1, 0, 0],
                                   [0, sm.cos(alpha), -sm.sin(alpha)],
                                   [0, sm.sin(alpha), sm.cos(alpha)]])


def R_y(alpha): return smm.Matrix([[sm.cos(alpha), 0, sm.sin(alpha)],
                                   [0, 1, 0],
                                   [-sm.sin(alpha), 0, sm.cos(alpha)]])


def R_z(alpha): return smm.Matrix([[sm.cos(alpha), -sm.sin(alpha), 0],
                                   [sm.sin(alpha), sm.cos(alpha), 0],
                                   [0, 0, 1]])


# total rotation
def R(a, b, c): return R_x(a)*R_y(b)*R_z(c)


# polarization vector function
def P(tp, pi, ti):
    p0 = smm.Matrix([1, 0, 0])
    R_i = R(0, -ti, -pi)*R_z(tp)
    return R_i * p0


# incident light
Ei = smm.Matrix([
    sm.cos(p)*sm.sin(t),
    sm.sin(p)*sm.sin(t),
    sm.cos(t)
])

# rotation matrix (VH)
R = smm.Matrix([[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]])

# both polarization components
Em_VV = Ei.copy()
Em_VH = R*Ei.copy()


# T2 tensor functs
def I_T2x(Em): return (Em.T * T_2x * Ei)**2


def I_T2y(Em): return (Em.T * T_2y * Ei)**2


def I_T2z(Em): return (Em.T * T_2z * Ei)**2


def I_T2(Em): return I_T2x(Em) + I_T2y(Em) + I_T2z(Em)


# E tensor
def I_E1(Em): return (Em.T * E1 * Ei)**2


def I_E2(Em): return (Em.T * E2 * Ei)**2


def I_E(Em): return sm.trigsimp(I_E1(Em) + I_E2(Em))


# A1 tensor
def I_A1(Em): return sm.trigsimp((Em.T * A_1 * Ei)**2)


# A1g tensor
def I_A1g(Em): return sm.trigsimp((Em.T * A1g * Ei)**2)


# E2g mode
def I_E2g1(Em): return (Em.T * E2g1 * Ei)**2


def I_E2g2(Em): return (Em.T * E2g2 * Ei)**2


def I_E2g(Em): return I_E2g1(Em) + I_E2g2(Em)


# Numpy type lambdas to calculate proper tensor representations
def IA1_np(Em, a_param, p_param, t_param):
    try:
        value = sm.lambdify([a], I_A1(Em), 'numpy')(a_param)[0][0]
        return np.array([[[value for p0 in p_param]]])
    except SyntaxError as e:
        print(Em, I_A1(Em))
        raise e


def IE_np(Em, b_param, p_param, t_param):
    return sm.lambdify([b, p, t], I_E(Em),
                       'numpy')(b_param, p_param, t_param)[0][0]


def IT2_np(Em, d_param, p_param, t_param):
    return sm.lambdify([d, p, t], I_T2(Em),
                       'numpy')(d_param, p_param, t_param)[0][0]


def IA1g_np(Em, a_param, b_param, p_param, t_param):
    return sm.lambdify([a, b, p, t],
                       I_A1g(Em), 'numpy')(a_param, b_param, p_param, t_param)


def IE2g_np(Em, d_param, p_param, t_param):
    return sm.lambdify([d, p, t], I_E2g(Em),
                       'numpy')(d_param, p_param, t_param)


# %%
phi0 = sm.Symbol(r'\phi_0')

Intensity = ((Em_VV.T*R_z(phi0)*T_2x*R_z(phi0).T*Ei)**2 +
             (Em_VV.T*R_z(phi0)*T_2y*R_z(phi0).T*Ei)**2 +
             (Em_VV.T*R_z(phi0)*T_2z*R_z(phi0).T*Ei)**2)

Intensity = sm.simplify(Intensity).subs({t: sm.pi/2})

# %%
