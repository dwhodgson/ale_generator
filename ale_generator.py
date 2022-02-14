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
from math import exp, pi, atan, sqrt
from typing import List
from csv import writer
from pickle import dump, HIGHEST_PROTOCOL


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
    with a slit of capacity `c` attached at the angle `theta` (i.e. the point e^(i*`theta`)). This is done
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
    for theta in thetas:
        if type(theta) != float:
            raise TypeError('Unexpected type for an element of parameter `thetas`: ' + str(type(theta)) + ', `thetas` '
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


def __length(c: float) -> float:
    """Computes the length of a slit with logarithmic capacity `c`.

    Parameters
    -----------
    c : float
        Logarithmic capacity of the slit.

    Returns
    -------
    float
        The length of a slit of logarithmic capacity `c`.

    """
    # Type checking
    if type(c) != float:
        raise TypeError('Unexpected type for parameter `c`: ' + str(type(c)) + ', `c` must be of type float')
    return 2 * exp(c) * (1 + sqrt(1 - exp(-c))) - 2


def __beta(c: float) -> float:
    """Computes the half-length of the sub-interval of [-pi,pi] for which points on the circle with argument within the
    sub-interval get mapped to the slit.

    Points on the circle that are mapped to points on the slit under a slit map compose a closed arc of the circle
    corresponding to the arguments in an interval [theta-b_c, theta+b_c] where b_c is the output of this function when
    `c` is the logarithmic capacity of the slit.

    Parameters
    -----------
    c : float
        The logarithmic capacity of the slit

    Returns
    -------
    float
        The half-length of the sub-interval.
    """
    # Type checking
    if type(c) != float:
        raise TypeError('Unexpected type for parameter `c`: ' + str(type(c)) + ', `c` must be of type float')
    return 2 * atan(__length(c) / (2 * sqrt(__length(c) + 1)))


def __generation(m: int, theta: float, caps: List[float], angles: List[float]) -> int:
    """Computes the generation number of the slit that a new slit attached at angle `theta` would attach to if it was
    the `m`+1th slit in the cluster generated by `caps` and `angles` up to generation m.

    The generation number of a slit is the time at which it was attached to the cluster. This function computes the
    generation number of the slit a new slit attached at angle `theta` would attach to at time `m`+1. This is done by
    checking whether `theta` falls into the interval of half-length beta_c which is taken to the `m`th slit under the
    `m`th map. If not, it recursively checks whether the pre-image of the angle under the `m`th building block map
    falls into the corresponding interval around the `m`-1th angle, etc.

    Parameters
    ----------
    m : int
        The generation up to which the cluster is built before the addition of a slit at `theta`.
    theta : float
        The angle at which the new slit would be attached.
    caps : List[float]
        The logarithmic capacities generating the cluster.
    angles : List[float]
        The angles generating the cluster. NB: named `angles` to avoid individual elements clashing with `theta`.

    Returns
    -------
    n : int
        The generation number of the slit a new slit attached at angle `theta` would attach to.

    """
    # Type checking
    if type(m) != int:
        raise TypeError('Unexpected type for parameter `m`: ' + str(type(m)) + ', `m` must be of type int')
    if type(theta) != float:
        raise TypeError('Unexpected type for parameter `theta`: ' + str(type(theta)) + ', `theta` must be of type '
                                                                                       'float')
    if type(caps) != list:
        raise TypeError('Unexpected type for parameter `caps`: ' + str(type(caps)) + ', `caps` must be a list of '
                                                                                     'floats')
    for cap in caps:
        if type(cap) != float:
            raise TypeError(
                'Unexpected type for an element of parameter `caps`: ' + str(type(cap)) + ', `caps` must be a list of '
                                                                                          'floats')
    if type(angles) != list:
        raise TypeError('Unexpected type for parameter `angles`: ' + str(type(angles)) + ', `angles` must be a list of '
                                                                                         'floats')
    for angle in angles:
        if type(angle) != float:
            raise TypeError('Unexpected type for an element of parameter `angles`: ' + str(type(angle)) + ', `angles` '
                                                                                                          'must be a '
                                                                                                          'list of '
                                                                                                          'floats')
    # Check angles are in the correct range
    for angle in angles:
        if angle < -pi or angle > pi:
            raise ValueError('Each angle should be in the range [-pi,pi]')
    if theta < -pi or theta > pi:
        raise ValueError('`theta` should be in the range [-pi,pi]')
    # Check length of lists
    if len(caps) != len(angles):
        raise ValueError('Lists `caps` and `angles` must be of the same length')
    for n in range(m, -1, -1):
        if n == 0:
            return 0
        angle = angles[n - 1]
        c = caps[n - 1]
        b = __beta(c)
        if angle - b <= theta <= angle + b:
            return n
        # Check in case of wrap-around
        if angle - b < -pi and theta >= angle - b + pi:
            return n
        if angle + b > pi and theta <= angle + b - pi:
            return n
        theta = cmath.phase(__building_block(c, cmath.exp(theta * 1j), angle))


def gaps(thetas: List[float], caps: List[float]) -> List[int]:
    """Computes the generation gaps for a cluster generated by `thetas` and `caps`.

    The generation gap of a slit is the difference between its generation number and the generation number of slit it is
    attached to. This function computes the generation gap for the slits corresponding to each angle in `thetas` for the
    cluster generated by `thetas` and `caps`.

    Parameters
    ----------
    thetas : List[float]
        The angles generating the cluster.
    caps : List[float]
        The logarithmic capacities generating the cluster. Should be of the same length as `thetas`.

    Returns
    -------
    List[float]
        The generation gaps for the cluster generated by `thetas` and `caps`.

    """
    # Type checking
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
    for theta in thetas:
        if type(theta) != float:
            raise TypeError('Unexpected type for an element of parameter `thetas`: ' + str(type(theta)) + ', `thetas` '
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
    return [(i + 1) - __generation(i, thetas[i], caps, thetas) for i in range(len(thetas))]


class LaplacianModel:
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
        self.thetas = []
        self.caps = []
        self.gaps = []
        self.name = None  # Overwritten by subclasses

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
        # Attribute checking
        if self.thetas == [] or self.caps == []:
            raise AttributeError('Angles or capacities (or both) have not been generated for this model')
        t = linspace(-pi, pi, self.points)
        x, y = sin(t) * self.reg, cos(t) * self.reg
        z = [complex(x_i, y_i) for x_i, y_i in zip(x, y)]
        z = [full_map(self.caps, z_i, self.thetas) for z_i in z]
        x, y = [z_i.real for z_i in z], [z_i.imag for z_i in z]
        plt.cla()
        plt.plot(x, y, ',')
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(self.name + '.png')

    def plot_gaps(self) -> None:
        """Plots the generation gaps as a histogram.

        Raises
        ------
        AttributeError
            If generation gaps have not been computed for this model"""
        # Attribute checking
        if self.gaps == []:
            raise AttributeError('Generation gaps have not been computed for this model')
        plt.cla()
        plt.hist(self.gaps)
        plt.savefig(self.name + '-GAPS.png')

    def save_gaps(self) -> None:
        """Saves the generation gaps as a csv file.

        Raises
        ------
        AttributeError
            If generation gaps have not been computed for this model

        """
        # Attribute checking
        if self.gaps == []:
            raise AttributeError('Generation gaps have not been computed for this model')
        with open(self.name + '-GAPS.csv', 'w') as file:
            gap_writer = writer(file, delimiter=',', quotechar='"')
            gap_writer.writerow(self.gaps)

    def compute_gaps(self, save: bool = True, plot: bool = True) -> None:
        """Computes the time series of generation gaps for the cluster given by generated angles and capacities.

        Computes the generation gaps, that is the difference between the generation of each particle and the generation
        of the particle it is attached to, by recursively removing relevant parts of a partition of the circle.

        Parameters
        ----------
        save : bool, default: True
            Whether to save the computed gaps as a csv file.
        plot : bool, default: True
            Whether to plot the gaps as a histogram.

        Raises
        ------
        AttributeError
            If angles or capacities have not been generated for this model.

        """
        # Type checking
        if type(save) != bool:
            raise TypeError('Unexpected type for parameter `save`: ' + str(type(save)) + '`save` must be of type bool')
        if type(plot) != bool:
            raise TypeError('Unexpected type for parameter `plot`: ' + str(type(plot)) + '`plot` must be fo type bool')
        # Attribute checking
        if self.thetas == [] or self.caps == []:
            raise AttributeError('Angles or capacities (or both) have not been generated for this model')
        self.gaps = gaps(self.thetas, self.caps)
        if save:
            self.save_gaps()
        if plot:
            self.plot_gaps()

    def save_angles_and_caps(self) -> None:
        """Saves generated angles and capacities as csv files.

        Raises
        ------
        AttributeError
            If angles or capacities have not been generated for this model.

        """
        # Attribute checking
        if self.thetas == [] or self.caps == []:
            raise AttributeError('Angles or capacities (or both) have not been generated for this model')
        with open(self.name + '-ANGLES.csv', 'w') as file:
            angle_writer = writer(file, delimiter=',', quotechar='"')
            angle_writer.writerow(self.thetas)
        with open(self.name + '-CAPS.csv', 'w') as file:
            cap_writer = writer(file, delimiter=',', quotechar='"')
            cap_writer.writerow(self.caps)

    def save_model(self) -> None:
        """Saves the model as a pickle file."""
        with open(self.name + '.pkl', 'w') as file:
            dump(self, file, HIGHEST_PROTOCOL)


class ALE(LaplacianModel):
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
