"""A module implementing off-lattice conformal Laplacian random growth models.

This module implements three different kinds of off-lattice Laplacian random growth models:
Aggregated Loewner Evolution, parameterised by alpha and eta.
Hastings-Levitov, parameterised by alpha; equivalent to ALE with eta=0.
The special case Hastings-Levitov-0; equivalent to ALE with alpha=eta=0.

Notes
------
For mathematical justification and explanation of the models implemented here see Sola, Turner, and Viklund [1]_ for ALE
and Hastings and Levitov [2]_ for HL.

References
----------
.. [1] A. Sola, A. Turner, F. Viklund, "One-Dimensional Scaling Limits in a Planar Laplacian Random Growth Model",
  Communications in Mathematical Physics, vol. 371, pp. 285-329, 2019.

.. [2] MB. Hastings, LS. Levitov, "Laplacian Growth as One-Dimensional Turbulence", Physica D, vol. 116, pp. 244-252,
  1998.

"""
import cmath
import random
import sys
import matplotlib.pyplot as plt
from numpy import linspace, sin, cos
from math import exp, pi, log10
from typing import List


def __slit_map(c: float, z: complex or float) -> complex:
    """Draws a slit with capacity `c` at the point 1.

    Computes the image of a point in the complex plane `z` under the map which takes the unit disc to the unit disc
    with a slit of capacity `c` at the point 1. Formula given by Loewner's equation.

    Parameters
    -----------
    c : float
        The logarithmic capacity of the slit.
    z : complex or float
        The point at which to compute the image of the slit map. If float computes the image at `z`+0*i.

    Returns
    -------
    complex
        The image of the point `z` under the slit map.

    """
    # Type checking
    if type(z) != complex:
        if type(z) == float:
            z = complex(z, 0)
        else:
            raise TypeError('Unexpected type for parameter `z`: ' + str(type(z)) + ', `z` must be of type complex '
                                                                                   'or float')
    if type(c) != float:
        raise TypeError('Unexpected type for parameter `c`:' + str(type(c)) + ', `c` must be of type float')
    return (exp(c) / (2 * z)) * (z ** 2 + 2 * z * (1 - exp(-c)) + 1 + (z + 1) ** 2 * cmath.sqrt(
        (z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1) / ((z + 1) ** 2)))


def __building_block(c: float, z: complex or float, theta: float) -> complex:
    """Draws a slit of capacity `c` attached at the angle `theta`

    Computes the image of a point in the complex plane `z` under the map which takes the unit disc to the unit disc
    with a slit of capacity `c` attached at the angle `theta` (i.e. the point :math: \exp{i`\theta`}). This is done
    by rotating through `theta`, attaching a slit at the point 1, then rotating back again.

    Parameters
    -----------
    c : float
        The logarithmic capacity of the slit.
    z : complex or float
        The point at which to compute the image of the map. If float computes the image at `z`+0*i.
    theta : float
        The angle at which to attach the slit to the unit disc - must be in the range [-pi,pi].

    Returns
    --------
    complex
        The image of the point `z` under the rotated slit map.
    """
    # Type checking
    if type(z) != complex:
        if type(z) == float:
            z = complex(z, 0)
        else:
            raise TypeError('Unexpected type for parameter `z`: ' + str(type(z)) + ', `z` must be of type complex '
                                                                                   'or float')
    if type(c) != float:
        raise TypeError('Unexpected type for parameter `c`:' + str(type(c)) + ', `c` must be of type float')
    if type(theta) != float:
        raise TypeError('Unexpected type for parameter `theta`:' + str(type(theta)) + ', `theta` must be of type '
                                                                                      'float')
    # Check theta is in the correct range
    if theta < -pi or theta > pi:
        raise ValueError('`theta` must be in the range [-pi,pi]')
    return cmath.exp(theta * 1j) * __slit_map(c, cmath.exp(-theta * 1j) * z)


def full_map(caps: List[float], z: complex or float, thetas: List[float]) -> complex:
    """Applies the full map of a Laplacian growth model.

    Computes the image of a point in the complex plane `z` under the full map of a Laplacian random growth model -
    that is, the backwards composition of len(thetas) building blocks.

    Parameters
    -----------
    caps : List[float]
       A list of the logarithmic capacity of each slit.
    z : complex or float
       The point at which to compute the image of the map. If float, computes the image at `z`+0*i.
    thetas : List[float]
        A list of the angles at which to attach each slit. Must be of the same length as `caps` and each element must
        be in the range [-pi,pi].

    Returns
    -------
    z : complex
        The image of the parameter `z` under the full map.
    """
    # Type checking
    if type(z) != complex:
        if type(z) == float:
            z = complex(z, 0)
        else:
            raise TypeError('Unexpected type for parameter `z`: ' + str(type(z)) + ', `z` must be of type complex '
                                                                                   'or float')
    if type(caps) != list:
        raise TypeError('Unexpected type for parameter `caps`: ' + str(type(caps)) + ', `caps` must be a list of '
                                                                                     'floats')
    for cap in caps:
        if type(cap) != float:
            raise TypeError(
                'Unexpected type for an element of parameter `caps`: ' + str(type(cap)) + ', `caps` must be a list of '
                                                                                          'floats')
    if type(thetas) != list:
        raise TypeError('Unexpected type for parameter `thetas`: ' + str(type(thetas)) + ', `thetas` must be a list of '
                                                                                         'floats')
    for cap in caps:
        if type(cap) != float:
            raise TypeError('Unexpected type for an element of parameter `thetas`: ' + str(type(thetas)) + ', `thetas` '
                                                                                                           'must be a '
                                                                                                           'list of '
                                                                                                           'floats')
    # Check thetas are in the correct range
    for theta in thetas:
        if theta < -pi or theta > pi:
            raise ValueError('Each angle should be in the range [-pi,pi]')
    # Check length of lists
    if len(caps) != len(thetas):
        raise ValueError('Lists `caps` and `thetas` must be of the same length')
    # Backwards composition of len(thetas) building blocks.
    for i in range(len(thetas) - 1, -1, -1):
        z = __building_block(caps[i], z, thetas[i])
    return z


class Laplacian_Model():
    """Super-Class for all implemented Laplacian growth models containing those parameters and methods shared by all
    models.

    This is a class designed to contain all the parameters and methods that are shared among the three implemented
    models.
    These include the number of iterations to be performed, the initial logarithmic capacity, etc. This class is used
    exclusively to subclass the individual models and should not be instantiated directly.

    Parameters
    -----------
    sigma : float
        Regularisation parameter. Maps and their derivatives will be applied to e^(i*theta + `sigma`).
    c : float
        The initial logarithmic capacity of the model.
    n : int
        The number of iterations to be performed.
    points : int, default: 10^6
        The number of points to be used for plotting clusters
    reg : float, default: 1.000001
        Regularisation of the circle for plotting clusters - should be strictly greater than 1.
    """

    def __init__(self, sigma: float, c: float, n: int, points: int = 10 ** 6, reg: float = 1.000001) -> None:
        self.sigma = sigma
        self.c = c
        self.n = n
        self.points = points
        self.reg = reg

    def plot(self) -> None:
        """Plots an image of the cluster given by generated angles and capacities.

        Plots the cluster given by previously generated angles and capacities by plotting the image of self.points
        equally spaced points around a circle using pyplot and saves the image as 'Name(alpha,eta,sigma,c,n).png' where
        Name is one of ALE, HL, or HL0 appropriately. Parameters are omitted where not relevant to the model, e.g. a
        Hastings-Levtiov-0 cluster would be saved as 'HL0(sigma,c,n).png'.

        Raises
        ------
        AttributeError
            If angles or capacities have not been generated for this model.

        """
        t = linspace(-pi, pi, self.points)
        x, y = sin(t) * self.reg, cos(t) * self.reg
        z = [complex(x_i,y_i) for x_i,y_i in zip(x,y)]
        z = full_map(self.caps,z,self.thetas)
        x,y = [z_i.real for z_i in z], [z_i.imag for z_i in z]
        plot_file = self.name + '.png'
        plt.cla()
        plt.plot(x, y, ',')
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(plot_file)


class ALE(Laplacian_Model):
    """
    TODO DOCSTRING
    """

    def __init__(self, alpha: float, eta: float, sigma: float, c: float, n: int, points: int = None, reg: float =
                 1.000001, bins: int = 10 ** 5) -> None:
        super().__init__(sigma, c, n, points, reg)
        self.alpha = alpha
        self.eta = eta
        self.bins = bins
        self.name = 'ALE(' + str(alpha) + ',' + str(eta) + ',' + str(sigma) + ',' + str(c) + ',' + str(n) + ')'
