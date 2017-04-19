"""
Provides a function to solve for asset prices under one set of beliefs in the
Harrison -- Kreps model.

Authors: Chase Coleman, Tom Sargent
"""
import numpy as np
import scipy.linalg as la

def price_singlebeliefs(transition, dividend_payoff, beta=.75):
    """
    Function to Solve Single Beliefs
    """
    # First compute inverse piece
    imbq_inv = la.inv(np.eye(transition.shape[0]) - beta*transition)

    # Next compute prices
    prices = beta * np.dot(np.dot(imbq_inv, transition), dividend_payoff)

    return prices
