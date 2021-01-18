# %%
import numpy as np
import matplotlib.pyplot as plt


def arc_2D(axes, center, vector, direction, *args, **kwargs):
    center = np.array(center)
    vector = np.array(vector)
    # getting angles
    angle_0 = np.arctan2(*(vector[::-1]))
    angle_1 = 0
    vec_norm = (vector.dot(vector))**0.5
    if '__len__' in dir(direction):
        if len(direction) == len(vector):
            direction = np.array(direction)
            angle_1 = np.arctan2(*(direction[::-1]))
            angle = angle_1 - angle_0
    else:
        angle = direction
        angle_1 = angle + angle_0
    # setting angle accuracy
    if (angle/100)/np.pi > 5/180:
        d_ang = 5/180 * np.pi
    else:
        d_ang = angle/100

    phi = np.arange(angle_0, angle_1+d_ang, d_ang)
    rad = np.linspace(vec_norm, vec_norm, len(phi))
    # TODO: split kwargs and args between text and plot
    # TODO: center text radius position based on angle
    if 'textpos' in kwargs.keys():
        textpos = kwargs.pop(textpos)
    else:
        angle_center = (angle_0+angle_1)/2
        rad_center = 3*vec_norm/4
        textpos = angle_center, rad_center

    if 'projection' in kwargs.keys():
        if kwargs.pop('projection') == 'polar':
            circle = [phi, rad]
        else:
            circle = [rad*np.cos(phi), rad*np.sin(phi)]
    else:
        angc, radc = textpos
        textpos = radc*np.cos(angc), radc*np.sin(angc)
        circle = [rad*np.cos(phi), rad*np.sin(phi)]

    if 'text' in kwargs.keys():
        txt = kwargs.pop('text')
    else:
        txt = r'$\theta$'
    axes.text(*textpos, txt)
    axes.plot(*circle, *args, **kwargs)
