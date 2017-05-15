r"""
Filename: lucastree.py

Reference: http://quant-econ.net/py/lucas_model.html

Solves the price function for the Lucas tree in a continuous state
setting, using piecewise linear approximation for the sequence of
candidate price functions.  The consumption endownment follows the log
linear AR(1) process

.. math::

    log y' = \alpha log y + \sigma \epsilon

where y' is a next period y and epsilon is an iid standard normal shock.
Hence

.. math::

    y' = y^{\alpha} * \xi,

where

.. math::

    \xi = e^(\sigma * \epsilon)

The distribution phi of xi is

.. math::

    \phi = LN(0, \sigma^2),

where LN means lognormal.

"""
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import fixed_quad
from quantecon import compute_fixed_point


class LucasTree:
    """
    Class to store parameters of a the Lucas tree model, a grid for the
    iteration step and some other helpful bits and pieces.

    Parameters
    ----------
    gamma : scalar(float)
        The coefficient of risk aversion in the household's CRRA utility
        function
    beta : scalar(float)
        The household's discount factor
    alpha : scalar(float)
        The correlation coefficient in the shock process
    sigma : scalar(float)
        The volatility of the shock process
    grid_size : int
        The size of the grid to use

    Attributes
    ----------
    gamma, beta, alpha, sigma, grid_size : see Parameters
    grid : ndarray
        Properties for grid upon which prices are evaluated
    phi : scipy.stats.lognorm
        The distribution for the shock process

    Examples
    --------
    >>> tree = LucasTree(gamma=2, beta=0.95, alpha=0.90, sigma=0.1)
    >>> price_vals = compute_lt_price(tree)

    """

    def __init__(self, 
            gamma=2, 
            beta=0.95, 
            alpha=0.90, 
            sigma=0.1, 
            grid_size=100):

        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma

        # == Set the grid interval to contain most of the mass of the
        # stationary distribution of the consumption endowment == #
        ssd = self.sigma / np.sqrt(1 - self.alpha**2)
        grid_min, grid_max = np.exp(-4 * ssd), np.exp(4 * ssd)
        self.grid = np.linspace(grid_min, grid_max, grid_size)
        self.grid_size = grid_size

        # == set up distribution for shocks == #
        self.phi = lognorm(sigma)
        self.draws = self.phi.rvs(500)

        # == h(y) = beta * int G(y,z)^(1-gamma) phi(dz) == #
        self.h = np.empty(self.grid_size)
        for i, y in enumerate(self.grid):
            self.h[i] = beta * np.mean((y**alpha * self.draws)**(1 - gamma))



## == Now the functions that act on a Lucas Tree == #

def lucas_operator(f, tree, Tf=None):
    """
    The approximate Lucas operator, which computes and returns the
    updated function Tf on the grid points.

    Parameters
    ----------
    f : array_like(float)
        A candidate function on R_+ represented as points on a grid
        and should be flat NumPy array with len(f) = len(grid)

    tree : instance of LucasTree
        Stores the parameters of the problem

    Tf : array_like(float)
        Optional storage array for Tf

    Returns
    -------
    Tf : array_like(float)
        The updated function Tf

    Notes
    -----
    The argument `Tf` is optional, but recommended. If it is passed
    into this function, then we do not have to allocate any memory
    for the array here. As this function is often called many times
    in an iterative algorithm, this can save significant computation
    time.

    """
    grid,  h = tree.grid, tree.h
    alpha, beta = tree.alpha, tree.beta
    z_vec = tree.draws

    # == turn f into a function == #
    Af = lambda x: np.interp(x, grid, f)  

    # == set up storage if needed == #
    if Tf is None:
        Tf = np.empty_like(f)

    # == Apply the T operator to f using Monte Carlo integration == #
    for i, y in enumerate(grid):
        Tf[i] = h[i] + beta * np.mean(Af(y**alpha * z_vec))

    return Tf


def compute_lt_price(tree, error_tol=1e-6, max_iter=500, verbose=0):
    """
    Compute the equilibrium price function associated with Lucas
    tree lt

    Parameters
    ----------
    tree : An instance of LucasTree
        Contains parameters

    error_tol, max_iter, verbose
        Arguments to be passed directly to
        `quantecon.compute_fixed_point`. See that docstring for more
        information

    Returns
    -------
    price : array_like(float)
        The prices at the grid points in the attribute `grid` of the
        object

    """
    # == simplify notation == #
    grid, grid_size = tree.grid, tree.grid_size
    gamma = tree.gamma

    # == Create storage array for compute_fixed_point. Reduces  memory
    # allocation and speeds code up == #
    Tf = np.empty(grid_size)

    # == Initial guess, just a vector of zeros == #
    f_init = np.zeros(grid_size)
    f = compute_fixed_point(lucas_operator, 
                f_init, 
                error_tol,
                max_iter, 
                verbose, 
                10,
                'iteration',
                tree, 
                Tf=Tf)

    price = f * grid**gamma

    return price
