"""
Filename: asset_pricing.py

Computes asset prices with a Lucas style discount factor when the endowment
obeys geometric growth driven by a finite state Markov chain.  That is,

.. math::
    d_{t+1} = g(X_{t+1}) d_t

where 

    * :math:`\{X_t\}` is a finite Markov chain with transition matrix P.

    * :math:`g` is a given positive-valued function

References
----------

    http://quant-econ.net/py/markov_asset.html

"""
import numpy as np
import quantecon as qe
from numpy.linalg import solve, eigvals


class AssetPriceModel:
    r"""
    A class that stores the primitives of the asset pricing model.

    Parameters
    ----------
    beta : scalar, float
        Discount factor
    mc : MarkovChain
        Contains the transition matrix and set of state values for the state
        process
    gamma : scalar(float)
        Coefficient of risk aversion
    g : callable
        The function mapping states to growth rates

    """
    def __init__(self, beta=0.96, mc=None, gamma=2.0, g=np.exp):
        self.beta, self.gamma = beta, gamma
        self.g = g

        # == A default process for the Markov chain == #
        if mc is None:
            self.rho = 0.9
            self.sigma = 0.02
            self.mc = qe.tauchen(self.rho, self.sigma, n=25)
        else:
            self.mc = mc

        self.n = self.mc.P.shape[0]

    def test_stability(self, Q):
        """
        Stability test for a given matrix Q.
        """
        sr = np.max(np.abs(eigvals(Q)))
        if not sr < 1 / self.beta:
            msg = "Spectral radius condition failed with radius = %f" % sr
            raise ValueError(msg)



def tree_price(ap):
    """
    Computes the price-dividend ratio of the Lucas tree.

    Parameters
    ----------
    ap: AssetPriceModel
        An instance of AssetPriceModel containing primitives

    Returns
    -------
    v : array_like(float)
        Lucas tree price-dividend ratio

    """
    # == Simplify names, set up matrices  == #
    beta, gamma, P, y = ap.beta, ap.gamma, ap.mc.P, ap.mc.state_values
    J = P * ap.g(y)**(1 - gamma)

    # == Make sure that a unique solution exists == #
    ap.test_stability(J)

    # == Compute v == #
    I = np.identity(ap.n)
    Ones = np.ones(ap.n)
    v = solve(I - beta * J, beta * J @ Ones)

    return v


def consol_price(ap, zeta):
    """
    Computes price of a consol bond with payoff zeta

    Parameters
    ----------
    ap: AssetPriceModel
        An instance of AssetPriceModel containing primitives

    zeta : scalar(float)
        Coupon of the console

    Returns
    -------
    p : array_like(float)
        Console bond prices

    """
    # == Simplify names, set up matrices  == #
    beta, gamma, P, y = ap.beta, ap.gamma, ap.mc.P, ap.mc.state_values
    M = P * ap.g(y)**(- gamma)

    # == Make sure that a unique solution exists == #
    ap.test_stability(M)

    # == Compute price == #
    I = np.identity(ap.n)
    Ones = np.ones(ap.n)
    p = solve(I - beta * M, beta * zeta * M @ Ones)

    return p


def call_option(ap, zeta, p_s, epsilon=1e-7):
    """
    Computes price of a call option on a consol bond.

    Parameters
    ----------
    ap: AssetPriceModel
        An instance of AssetPriceModel containing primitives

    zeta : scalar(float)
        Coupon of the console

    p_s : scalar(float)
        Strike price

    epsilon : scalar(float), optional(default=1e-8)
        Tolerance for infinite horizon problem

    Returns
    -------
    w : array_like(float)
        Infinite horizon call option prices

    """
    # == Simplify names, set up matrices  == #
    beta, gamma, P, y = ap.beta, ap.gamma, ap.mc.P, ap.mc.state_values
    M = P * ap.g(y)**(- gamma)

    # == Make sure that a unique consol price exists == #
    ap.test_stability(M)

    # == Compute option price == #
    p = consol_price(ap, zeta)
    w = np.zeros(ap.n)
    error = epsilon + 1
    while error > epsilon:
        # == Maximize across columns == #
        w_new = np.maximum(beta * M @ w, p - p_s)
        # == Find maximal difference of each component and update == #
        error = np.amax(np.abs(w-w_new))
        w = w_new

    return w
