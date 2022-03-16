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
import ctypes
import cmath
import random
import matplotlib
import matplotlib.pyplot as plt
from numpy import linspace, sin, cos
from math import exp, pi, atan, sqrt
from typing import List
from csv import writer
from pickle import dump, HIGHEST_PROTOCOL

matplotlib.use('Agg')
plt.ioff()
# Uncomment for reproducibility
random.seed(72)

# C Functions
ale_funcs = ctypes.CDLL('./ale_funcs.so')
full_map_list = ale_funcs.full_map_py
full_map_list.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                          ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
simpson = ale_funcs.simpson
simpson.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double,
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_int,
                    ctypes.c_int]


def slit_map(c: float, z: complex) -> complex:
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


def building_block(c: float, z: complex, theta: float) -> complex:
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
    return cmath.exp(theta * 1j) * slit_map(c, cmath.exp(-theta * 1j) * z)


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
        z = building_block(caps[i], z, thetas[i])
    return z


def slit_diff(c: float, z: complex) -> complex:
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
    complex
        The derivative of the slit map with capacity `c` at the point `z`.

    """
    sq_rt = cmath.sqrt((z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1) / ((z + 1) ** 2))
    prod1 = -(exp(c) / (2 * z ** 2)) * (z ** 2 + 2 * z * (1 - exp(-c)) + 1 + (z + 1) ** 2 * sq_rt)
    prod2 = (exp(c) / (2 * z)) * (2 * z + 2 * (1 - exp(-c)) + 2 * (z + 1) * sq_rt + (z + 1) ** 2 * 0.5 * (1 / sq_rt) * (
            ((2 * z + 2 * (1 - 2 * exp(-c))) * (z + 1) ** 2 - 2 * (z + 1) * (z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1)
             ) / ((z + 1) ** 4)))
    return prod1 + prod2


def map_diff(z: complex, caps: List[float], thetas: List[float]) -> complex:
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
    diff : complex
        The value of the derivative of the map characterised by `caps` and `thetas` at the point e^(i*`theta` +
        `sigma`).
    """
    diff = 1
    for i in range(len(caps) - 1, -1, -1):
        diff = diff * slit_diff(caps[i], cmath.exp(-thetas[i] * 1j) * z)
        z = building_block(caps[i], z, thetas[i])
    return diff


def length(c: float) -> float:
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


def beta(c: float) -> float:
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
    return 2 * atan(length(c) / (2 * sqrt(length(c) + 1)))


def generation(m: int, theta: float, caps: List[float], angles: List[float]) -> int:
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
        b = beta(c)
        if angle - b <= theta <= angle + b:
            return n
        # Check in case of wrap-around
        if angle - b < -pi and theta >= angle - b + 2 * pi:
            return n
        if angle + b > pi and theta <= angle + b - 2 * pi:
            return n
        theta = cmath.phase(building_block(c, cmath.exp(theta * 1j + 0.00000000001), angle))


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
    List[int]
        The generation gaps for the cluster generated by `thetas` and `caps`.

    """
    return [(i + 1) - generation(i, thetas[i], caps, thetas) for i in range(len(thetas))]


def plot(caps: List[float], thetas: List[float], filename: str = 'Cluster.png', points: int = 10 ** 6,
         reg: float = 1.0000001, display: bool = True) -> None:
    """Plots the cluster generated by a given list of angles and capacities.

    Plots the cluster generated by the capacities and angles given in `caps` and `thetas`. this is done by applying
    the full map to `points` points evenly distributed around a circle distance 1-`reg` away from the unit circle. This
    plot is then saved under the name given in `filename` and possibly displayed.

    Parameters
    ----------
    caps : List[float]
        The logarithmic capacities generating the cluster.
    thetas : List[float]
        The angles generating the cluster. Should be of the same length as `caps` and each element should be in the
        range [-pi,pi].
    filename : str, default: 'Cluster.png'
        The name under which to save the plot.
    points : int, default: 10^6
        The number of points used to plot the cluster.
    reg : float, default: 1.0000001
        The regularisation to use for points when plotting.
    display : bool, default: True
        Whether to display the plot as well as saving.

    """
    t = linspace(-pi, pi, points)
    x, y = sin(t) * reg, cos(t) * reg
    PointArray = ctypes.c_double * points
    xc, yc = PointArray(*x), PointArray(*y)
    DoubleArray = ctypes.c_double * len(thetas)
    cap_list = DoubleArray(*caps)
    theta_list = DoubleArray(*thetas)
    full_map_list(cap_list, xc, yc, theta_list, len(thetas), points)
    plt.cla()
    plt.plot(xc, yc, ',')
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if display:
        plt.show()
    plt.savefig(filename)


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

    def plot(self, display: bool = False) -> None:
        """Plots an image of the cluster given by generated angles and capacities.

        Plots the cluster given by previously generated angles and capacities by plotting the image of self.points
        equally spaced points around a circle using pyplot and saves the image as 'Name(alpha,eta,sigma,c,n).png' where
        Name is one of ALE, HL, or HL0 appropriately. Parameters are omitted where not relevant to the model, e.g. a
        Hastings-Levtiov-0 cluster would be saved as 'HL0(sigma,c,n).png'.

        Parameters
        ----------
        display : bool, default: False
            Whether to display the plot to the user as well as saving.

        Raises
        ------
        AttributeError
            If angles or capacities have not been generated for this model.

        """
        # Attribute checking
        if len(self.thetas) == 0 or len(self.caps) == 0:
            raise AttributeError('Angles or capacities (or both) have not been generated for this model')
        plot(self.caps, self.thetas, self.name + '.png', self.points, self.reg, display)

    def plot_colour(self, display: bool = False) -> None:
        """Plots an image of the cluster given by generated angles and capacities, with slit generation marked by
        colour.

        Plots the cluster given by previously generated angles and capacities by plotting the image of self.points
        equally spaced points around a circle using pyplot with the slits' generations marked by colour - earlier
        slits are plotted in red with later slits moving through green to blue. The image is saved as 'Name(alpha,
        eta,sigma,c,n)-COLOUR.png' where Name is one of ALE, HL, or HL0 appropriately. Parameters are omitted where
        not relevant to the model, e.g. a Hastings-Levtiov-0 cluster would be saved as 'HL0(sigma,c,n)-COLOUR.png'.

        Parameters
        ----------
        display : bool, default: False
            Whether to display the plot to the user as well as saving.

        Raises
        ------
        AttributeError
            If angles or capacities have not been generated for this model.

        """
        # Attribute checking
        if len(self.caps) == 0 or len(self.thetas) == 0:
            raise AttributeError('Angles or capacities (or both) have not been generated for this model')
        t = linspace(-pi, pi, self.points)
        x, y = sin(t) * self.reg, cos(t) * self.reg
        PointArray = ctypes.c_double * self.points
        xc, yc = PointArray(*x), PointArray(*y)
        DoubleArray = ctypes.c_double * self.n
        cap_list = DoubleArray(*self.caps)
        theta_list = DoubleArray(*self.thetas)
        full_map_list(cap_list, xc, yc, theta_list, self.n, self.points)
        colour = (0, 0, 1)
        switch = len(self.thetas) // 511
        plt.cla()
        plt.plot(xc, yc, ',', color=colour)
        for i in range(1, len(self.thetas), 1):
            xc, yc = PointArray(*x), PointArray(*y)
            DoubleArray = ctypes.c_double * (self.n - i)
            cap_list = DoubleArray(*self.caps[:-i])
            theta_list = DoubleArray(*self.thetas[:-i])
            full_map_list(cap_list, xc, yc, theta_list, self.n - i, self.points)
            if (i + 1) % switch == 0:
                if colour[2] != 0:
                    colour = (0, colour[1] + 1 / 255, colour[2] - 1 / 255)
                else:
                    colour = (colour[0] + 1 / 255, colour[1] - 1 / 255, 0)
            plt.plot(xc, yc, ',', color=colour)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if display:
            plt.show()
        plt.savefig(self.name + '-COLOUR.png')

    def plot_gaps(self, display: bool = False) -> None:
        """Plots the generation gaps as a histogram.

        Plots the generation gaps produced by compute_gaps() as a pyplot histogram. The resulting image is saved as
        'Name(alpha,eta,sigma,c,n)-GAPS.png' where Name is one of ALE, HL, or HL0 appropriately. Parameters are
        omitted where not relevant to the model, e.g. a Hastings-Levtiov-0 cluster would be saved as 'HL0(sigma,c,
        n)-GAPS.png.'

        Parameters
        ----------
        display : bool, default: False
            Whether to display the plot to the user as well as saving.

        Raises
        ------
        AttributeError
            If generation gaps have not been computed for this model"""
        # Attribute checking
        if len(self.gaps) == 0:
            raise AttributeError('Generation gaps have not been computed for this model')
        plt.cla()
        plt.hist(self.gaps)
        if display:
            plt.show()
        plt.savefig(self.name + '-GAPS.png')

    def save_gaps(self) -> None:
        """Saves the generation gaps as a csv file.

        Saves the generation gaps produced by compute_gaps() as a csv file. The list is saved as 'Name(alpha,eta,
        sigma,c,n)-GAPS.csv' where Name is one of ALE, HL, or HL0 appropriately. Parameters are omitted where not
        relevant to the model, e.g. a Hastings-Levtiov-0 cluster would be saved as 'HL0(sigma,c, n)-GAPS.csv.'

        Raises
        ------
        AttributeError
            If generation gaps have not been computed for this model

        """
        # Attribute checking
        if len(self.gaps) == 0:
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
        if len(self.thetas) == 0 or len(self.caps) == 0:
            raise AttributeError('Angles or capacities (or both) have not been generated for this model')
        self.gaps = gaps(self.caps, self.thetas)
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
        if len(self.thetas) == 0 or len(self.caps) == 0:
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
    """A class implementing the Aggregated Loewner Evolution model of random growth.

    This class implements the most complex of the three contained in this module: Aggregated Loewner Evolution. This
    model is parameterised by two main parameters: `alpha`, which controls how the logarithmic capacities evolve, and
    `eta`, which controls the distribution from which attachment angles are chosen. Hastings-Levitov is equivalent to
    this model with `eta`=0 and HL0 is equivalent to this model with `eta`=`alpha`=0. See [1]_ for mathematical
    explanation and justification of this model.

    Parameters
    ----------
    alpha : float
        The parameter characterising the evolution of the logarithmic capacities.
    eta : float
        The parameter characterising the evolution of the attachment angle distribution.
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
    bins : int, default: 10^5
        The number of bins to be used when sampling from the angle distribution using Simpson's rule.

    References
    ----------
    .. [1] A. Sola, A. Turner, F. Viklund, "One-Dimensional Scaling Limits in a Planar Laplacian Random Growth Model",
    Communications in Mathematical Physics, vol. 371, pp. 285-329, 2019.

    """

    def __init__(self, alpha: float, eta: float, sigma: float, c: float, n: int, points: int = 10 ** 6,
                 reg: float = 1.000001, bins: int = 10 ** 5) -> None:
        super(ALE, self).__init__(sigma, c, n, points, reg)
        self.alpha = alpha
        self.eta = eta
        self.bins = bins
        self.name = 'ALE(' + str(alpha) + ',' + str(eta) + ',' + str(sigma) + ',' + str(c) + ',' + str(n) + ')'

    def pdf(self, theta: float) -> float:
        """Returns the un-normalised value of the pdf of the angle distribution at the angle `theta`.

        Parameters
        ----------
        theta : float
            The point at which to evaluate the pdf.

        Returns
        -------
        float
            The value of the pdf of the angle distribution at the point `theta`.

        """
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
        bins = linspace(-pi, pi, self.bins + 1)
        simpsons = [0 for _ in range(self.bins)]
        SimpsonArray = ctypes.c_double * self.bins
        simpsons = SimpsonArray(*simpsons)
        BinArray = ctypes.c_double * (self.bins + 1)
        bins = BinArray(*bins)
        DoubleArray = ctypes.c_double * len(self.thetas)
        theta_list = DoubleArray(*self.thetas)
        cap_list = DoubleArray(*self.caps)
        simpson(simpsons, bins, self.sigma, cap_list, theta_list, self.eta, len(self.thetas), self.bins)
        # simpsons = [((bins[i+1]-bins[i])/6)*(self.pdf(bins[i])+self.pdf(bins[i+1]) + 4*self.pdf((bins[i]+bins[i+1])/2)) for i in range(self.bins)]
        bins = [(bins[i + 1] + bins[i]) / 2 for i in range(self.bins)]
        total = sum(simpsons)
        rand = random.random() * total
        target = 0
        for i in range(len(bins)):
            target += simpsons[i]
            if rand < target:
                return bins[i]

    def generate_angles_and_capacities(self) -> None:
        """Generates the angles and capacities for this model.

        Generates the sequences of angles and capacities defined by the ALE model with the given parameters sigma,
        alpha, and eta which are stored in self.thetas and self.caps respectively. This is done by iteratively sampling
        a new angle from the distribution defined by the derivative of the map raised to the power -eta and then
        computing the next capacity by evaluating the derivative of the map at the newly sampled angle (regularised by
        sigma).

        """
        self.thetas.append(random.random() * 2 * pi - pi)
        self.caps.append(self.c)
        for _ in range(self.n - 1):
            new_angle = self.sample()
            diff = abs(map_diff(cmath.exp(self.sigma + new_angle * 1j), self.caps, self.thetas))
            self.caps.append(self.c * (diff ** -self.alpha))
            self.thetas.append(new_angle)


class HL(LaplacianModel):
    """
    TODO DOCSTRING
    """

    def __init__(self, alpha: float, sigma: float, c: float, n: int, points: int = 10 ** 6,
                 reg: float = 1.000001) \
            -> None:
        super(HL, self).__init__(sigma, c, n, points, reg)
        self.alpha = alpha
        self.name = 'HL(' + str(alpha) + ',' + str(sigma) + ',' + str(c) + ',' + str(n) + ')'
        self.thetas = [random.random() * 2 * pi for _ in range(n)]

    def generate_capacities(self) -> None:
        """Generates the sequence of capacities for this model.

        Generates the sequence of logarithmic capacities using the derivative of the slit map. Each derivative is
        computed at the point of attachment of the next slit.
        """
        self.caps.append(self.c)
        for i in range(self.n - 1):
            self.caps.append(self.c * abs(map_diff(cmath.exp(1j * self.thetas[i + 1] + self.sigma), self.caps,
                                                   self.thetas[:i + 1])) ** (-self.alpha))


class HL0(LaplacianModel):
    """
    TODO DOCSTRING
    """

    def __init__(self, sigma: float, c: float, n: int, points: int = 10 ** 6, reg: float = 1.000001) -> None:
        super(HL0, self).__init__(sigma, c, n, points, reg)
        self.name = 'HL0(' + str(sigma) + ',' + str(c) + ',' + str(n) + ')'
        self.caps = [c for _ in range(n)]
        self.thetas = [random.random() * 2 * pi - pi for _ in range(n)]
