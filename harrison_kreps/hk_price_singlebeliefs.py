"""

Authors: Chase Coleman, Tom Sargent

"""
import numpy as np
import scipy.linalg as la

def price_single_beliefs(transition, dividend_payoff, β=.75):
    """
    Function to Solve Single Beliefs
    """
    # First compute inverse piece
    imbq_inv = la.inv(np.eye(transition.shape[0]) - β*transition)

    # Next compute prices
    prices = β * np.dot(np.dot(imbq_inv, transition), dividend_payoff)

    return prices
