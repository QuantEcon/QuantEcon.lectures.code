"""
Author: Chase Coleman
Date: Today

This file is some simple calculations for Tom
"""
import numpy as np
import scipy.linalg as la
from hk_price_optimisticbeliefs import price_optimisticbeliefs
from hk_price_singlebeliefs import price_singlebeliefs
from hk_price_pessimisticbeliefs import price_pessimisticbeliefs


# ------------------------------------------------------------------- #
# Set Up Parameters
# ------------------------------------------------------------------- #
beta = .75
dividendreturn = np.array([[0], [1]])
qa = np.array([[1./2, 1./2], [2./3, 1./3]])
qb = np.array([[2./3, 1./3], [1./4, 3./4]])
qpess = np.array([[2./3, 1./3], [2./3, 1./3]])
qopt = np.array([[1./2, 1./2], [1./4, 3./4]])
qs_names = ["Qa", "Qb", "Qpess", "Qopt"]
the_qs = [qa, qb, qpess, qopt]


class PriceHolder(object):
    """
    This holds the results for Harrison Kreps.  In particular, it
    accepts two matrices Qa and Qb and compares the single belief,
    optimistic belief, and pessimistic belief prices
    """
    def __init__(self, qa, qb, dividend_payoff, beta=.75):
        # Unpack the parameters
        self.qa, self.qb = qa, qb
        self.dividend_payoff = dividend_payoff
        self.beta = .75
        self.max_iters = 10000
        self.tolerance = 1e-16

        # Create the Pessimistic and Optimistic Beliefs
        self.qpess = np.empty((2, 2))
        self.qpess[0, :] = qa[0, :] if qa[0, 1] < qb[0, 1] else qb[0, :]
        self.qpess[1, :] = qa[1, :] if qa[1, 1] < qb[1, 1] else qb[1, :]
        self.qopt = np.empty((2, 2))
        self.qopt[0, :] = qa[0, :] if qa[0, 1] > qb[0, 1] else qb[0, :]
        self.qopt[1, :] = qa[1, :] if qa[1, 1] > qb[1, 1] else qb[1, :]

        # Price everything
        self.create_prices()

    def __repr__(self):
        ret_str = "The Single Belief Price Vectors are:\n"+\
                  "P(Qa) = {}\nP(Qb) = {}\nP(Qopt) = {}\nP(Qpess) = {}\n\n"+\
                  "The Optimistic Belief Price Vector is:\n"+\
                  "P(Optimistic) = {}\n\n"+\
                  "Phat(a) = {}\n"+\
                  "Phat(b) = {}\n"+\
                  "The Pessimistic Belief Price Vector is:\n"+\
                  "P(Pessimistic) = {}"

        qaprice, qbprice, qpessprice, qoptprice = map(np.squeeze, [self.qaprice, self.qbprice, self.qpessprice, self.qoptprice])
        optimisticprice, pessimisticprice = map(np.squeeze, [self.optimisticprice, self.pessimisticprice])
        phata, phatb = map(np.squeeze, [self.phat_a, self.phat_b])

        return ret_str.format(qaprice, qbprice, qoptprice,
                              qpessprice, optimisticprice, phata, phatb,
                              pessimisticprice)

    def create_prices(self):
        """
        Computes prices under all belief systems
        """
        transitionmatrix = [self.qa, self.qb, self.qpess, self.qopt]

        # Single Belief Prices
        p_singlebelief = [price_singlebeliefs(q, self.dividend_payoff) for
                          q in transitionmatrix]

        # Compute Optimistic and Pessimistic beliefs
        p_optimistic, phat_a, phat_b = price_optimisticbeliefs([qa, qb], self.dividend_payoff)
        p_pessimistic = price_pessimisticbeliefs([qa, qb], self.dividend_payoff)

        self.qaprice = p_singlebelief[0]
        self.qbprice = p_singlebelief[1]
        self.qpessprice = p_singlebelief[2]
        self.qoptprice = p_singlebelief[3]
        self.phat_a = phat_a
        self.phat_b = phat_b

        self.optimisticprice = p_optimistic
        self.pessimisticprice = p_pessimistic

        return p_singlebelief, p_optimistic, p_pessimistic


ph = PriceHolder(qa, qb, dividendreturn)

print(ph)


##### Problems start here

ea = la.eig(qa)

eb = la.eig(qb)


print("ea =")
print(ea)

print("eb=")
print(eb)


eaa = np.linalg.matrix_power(qa, 100)


print("100th power of qa")
print(eaa)


ebb = np.linalg.matrix_power(qb, 100)


print("100th power of qb")
print(ebb)

import quantecon as qe


qa = np.array([[1./2, 1./2], [2./3, 1./3]])
qb = np.array([[2./3, 1./3], [1./4, 3./4]])

mcA = qe.MarkovChain(qa)
mcB = qe.MarkovChain(qb)

ppa = mcA.stationary_distributions
ppb = mcB.stationary_distributions

print("stationary distribution of P_a")

print(ppa)



mcB = qe.MarkovChain(qb)

ppb = mcB.stationary_distributions

print("stationary distribution of P_b")

print(ppb)
