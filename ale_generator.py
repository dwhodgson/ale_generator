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


def __slit_map(c: float, z: complex) -> complex:
    """Draws a slit with capacity `c` at the point 1.

    Computes the image of a point in the complex plane `z` under the map which takes the unit disc to the unit disc
    with a slit of capacity `c` at the point 1. Formula given by Loewner's equation.

    Parameters
    -----------
    c : float
        The logarithmic capacity of the slit.
    z : complex  float
        The point at which to compute the image of the slit map.

    Returns
    -------
    complex
        The image of the point `z` under the slit map.

    """
    return (exp(c) / (2 * z)) * (z ** 2 + 2 * z * (1 - exp(-c)) + 1 + (z + 1) ** 2 * cmath.sqrt(
        (z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1) / ((z + 1) ** 2)))


def __building_block(c: float, z: complex, theta: float) -> complex:
    """Draws a slit of capacity `c` attached at the angle `theta`

    Computes the image of a point in the complex plane `z` under the map which takes the unit disc to the unit disc
    with a slit of capacity `c` attached at the angle `theta` (i.e. the point e^(i*`theta`)). This is done
    by rotating through `theta`, attaching a slit at the point 1, then rotating back again.

    Parameters
    -----------
    c : float
        The logarithmic capacity of the slit.
    z : complex
        The point at which to compute the image of the map.
    theta : float
        The angle at which to attach the slit to the unit disc - should be in the range [-pi,pi].

    Returns
    --------
    complex
        The image of the point `z` under the rotated slit map.
    """
    return cmath.exp(theta * 1j) * __slit_map(c, cmath.exp(-theta * 1j) * z)


def full_map(caps: List[float], z: complex, thetas: List[float]) -> complex:
    """Applies the full map of a Laplacian growth model.

    Computes the image of a point in the complex plane `z` under the full map of a Laplacian random growth model -
    that is, the backwards composition of len(thetas) building blocks.

    Parameters
    -----------
    caps : List[float]
       A list of the logarithmic capacity of each slit.
    z : complex
       The point at which to compute the image of the map.
    thetas : List[float]
        A list of the angles at which to attach each slit. Should be of the same length as `caps` and each element
        should be in the range [-pi,pi].

    Returns
    -------
    z : complex
        The image of the parameter `z` under the full map.
    """
    # Backwards composition of len(thetas) building blocks.
    for i in range(len(thetas) - 1, -1, -1):
        z = __building_block(caps[i], z, thetas[i])
    return z


def __slit_diff(c: float, z: complex):
    """Computes the derivative of the slit map with capacity `c` at the point `z`.

    Calculates the derivative of the slit map with logarithmic capacity `c` at a point in the complex plane `z` via the
    product rule. This derivative is used iteratively to compute the derivative of a full map.

    Parameters
    ----------
    c : float
        The logarithmic capacity of the slit.
    z : complex
        The point in the complex plane at which to compute the derivative.

    Returns
    -------
    float
        The derivative of the slit map with capacity `c` at the point `z`.

    """
    sq_rt = cmath.sqrt((z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1) / ((z + 1) ** 2))
    prod1 = -(exp(c) / (2 * z ** 2)) * (z ** 2 + 2 * z * (1 - exp(-c)) + 1 + (z + 1) ** 2 * sq_rt)
    prod2 = (exp(c) / (2 * z)) * (2 * z + 2 * (1 - exp(-c)) + 2 * (z + 1) * sq_rt + (z + 1) ** 2 * 0.5 * (1 / sq_rt) * (
            ((2 * z + 2 * (1 - 2 * exp(-c))) * (z + 1) ** 2 - 2 * (z + 1) * (z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1)
             ) / ((z + 1) ** 4)))
    return prod1 + prod2


def map_diff(z: complex, caps: List[float], thetas: List[float]) -> float:
    """Computes the derivative of the full map characterised by `caps` and `thetas` at the point `z`.

    Calculates the derivative of the full map characterised by `caps` and `thetas` at the point `z` using the chain
    rule.

    Parameters
    ----------
    z : complex
        The point at which to compute the derivative.
    caps : List[float]
        The logarithmic capacities characterising the map.
    thetas : List[float]
        The angles characterising the map. Should be of the same length as `caps` and each element should be in the
        range [-pi,pi].

    Returns
    --------
    diff : float
        The value of the derivative of the map characterised by `caps` and `thetas` at the point e^(i*`theta` +
        `sigma`).
    """
    diff = 1
    for i in range(len(caps) - 1, -1, -1):
        diff = diff * __slit_diff(caps[i], cmath.exp(-thetas[i] * 1j) * z)
        z = __building_block(caps[i], z, thetas[i])
    return diff


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
        The angle at which the new slit would be attached. Should be in the range [-pi,pi]
    caps : List[float]
        The logarithmic capacities generating the cluster.
    angles : List[float]
        The angles generating the cluster - should be of the same length as `caps` and each element should be in the
        range [-pi,pi] NB: named `angles` to avoid individual elements clashing with `theta`.

    Returns
    -------
    n : int
        The generation number of the slit a new slit attached at angle `theta` would attach to.

    """
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


def gaps(caps: List[float], thetas: List[float]) -> List[int]:
    """Computes the generation gaps for a cluster generated by `thetas` and `caps`.

    The generation gap of a slit is the difference between its generation number and the generation number of slit it is
    attached to. This function computes the generation gap for the slits corresponding to each angle in `thetas` for the
    cluster generated by `thetas` and `caps`.

    Parameters
    ----------
    caps : List[float]
        The logarithmic capacities generating the cluster.
    thetas : List[float]
        The angles generating the cluster. Should be of the same length as `caps` and each element should be in the
        range [-pi,pi].

    Returns
    -------
    List[float]
        The generation gaps for the cluster generated by `thetas` and `caps`.

    """
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
        self.caps = [c]
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

    def __init__(self, alpha: float, eta: float, sigma: float, c: float, n: int, points: int = 10 ** 6,
                 reg: float = 1.000001, bins: int = 10 ** 5) -> None:
        super().__init__(sigma, c, n, points, reg)
        self.alpha = alpha
        self.eta = eta
        self.bins = bins
        self.name = 'ALE(' + str(alpha) + ',' + str(eta) + ',' + str(sigma) + ',' + str(c) + ',' + str(n) + ')'

    def __pdf(self, theta: float) -> float:
        return abs(map_diff(cmath.exp(self.sigma + theta * 1j), self.caps, self.thetas)) ** -self.eta

    def sample(self) -> float:
        """Samples an angle from the angle distribution using Simpson's rule.

        Samples an angle from the distribution with pdf proportional to the absolute value of the derivative of the
        slit map raised to the power of -eta. This is done by discretising the interval [-pi,pi] into the number of
        bins set by self.bins and selecting a midpoint weighted according to Simpson's rule.

        Returns
        -------
        float
            An angle in the interval [-pi,pi] randomly sampled from the requisite distribution.
        """
        bins = linspace(-pi, pi, self.bins)
        simpsons = [((bins[i+1]-bins[i])/6)*(self.__pdf(bins[i]) + 4 * self.__pdf((bins[i+1]-bins[i])/2) +
                                             self.__pdf(bins[i+1])) for i in range(len(bins)-1)]
        bins = [(bins[i+1]-bins[i])/2 for i in range(len(bins)-1)]
        return random.choices(bins, simpsons, k=1)[0]


class HL(LaplacianModel):
    """
    TODO DOCSTRING
    """

    def __init__(self, alpha: float, sigma: float, c: float, n: int, points: int = 10 ** 6,
                 reg: float = 1.000001) \
            -> None:
        super().__init__(sigma, c, n, points, reg)
        self.alpha = alpha
        self.name = 'HL(' + str(alpha) + ',' + str(sigma) + ',' + str(c) + ',' + str(n) + ')'
        self.thetas = [random.random() * 2 * pi for _ in range(n)]

    def generate_capacities(self) -> None:
        """Generates the sequence of capacities for this model.

        Generates the sequence of logarithmic capacities using the derivative of the slit map. Each derivative is
        computed at the point of attachment of the next slit.
        """
        for i in range(self.n - 1):
            self.caps.append(self.c * abs(map_diff(cmath.exp(1j * self.thetas[i + i] + self.sigma), self.caps,
                                                   self.thetas[:i + 1])) ** (-self.alpha))


class HL0(LaplacianModel):
    """
    TODO DOCSTRING
    """

    def __init__(self, sigma: float, c: float, n: int, points: int = 10 ** 6, reg: float = 1.000001) -> None:
        super().__init__(sigma, c, n, points, reg)
        self.name = 'HL0(' + str(sigma) + ',' + str(c) + ',' + str(n) + ')'
        self.caps = [c for _ in range(n)]
        self.thetas = [random.random() * 2 * pi - pi for _ in range(n)]
