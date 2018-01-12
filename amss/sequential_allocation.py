"""

@author: dgevans

"""
import numpy as np
from scipy.optimize import root
from scipy.optimize import fmin_slsqp
from scipy.interpolate import UnivariateSpline
from quantecon import compute_fixed_point, MarkovChain


class SequentialAllocation:
    '''
    Class returns planner's allocation as a function of the multiplier on the
    implementability constraint μ
    '''

    def __init__(self, model):
        '''
        Initializes the class from the calibration model
        '''
        self.β, self.π, self.G = model.β, model.π, model.G
        self.mc = MarkovChain(self.π)
        self.S = len(model.π)  # number of states
        self.Θ = model.Θ
        self.model = model
        # now find the first best allocation
        self.find_first_best()

    def find_first_best(self):
        '''
        Find the first best allocation
        '''
        model = self.model
        S, Θ, Uc, Un, G = self.S, self.Θ, model.Uc, model.Un, self.G

        def res(z):
            c = z[:S]
            n = z[S:]
            return np.hstack(
                [Θ * Uc(c, n) + Un(c, n), Θ * n - c - G]
            )
        res = root(res, 0.5 * np.ones(2 * S))

        if not res.success:
            raise Exception('Could not find first best')

        self.cFB = res.x[:S]
        self.nFB = res.x[S:]
        # multiplier on the resource constraint.
        self.ΞFB = Uc(self.cFB, self.nFB)
        self.zFB = np.hstack([self.cFB, self.nFB, self.ΞFB])

    def time1_allocation(self, μ):
        '''
        Computes optimal allocation for time t\geq 1 for a given \mu
        '''
        model = self.model
        S, Θ, G, Uc, Ucc, Un, Unn = self.S, self.Θ, self.G, model.Uc, model.Ucc, model.Un, model.Unn

        def FOC(z):
            c = z[:S]
            n = z[S:2 * S]
            Ξ = z[2 * S:]
            return np.hstack([
                Uc(c, n) - μ * (Ucc(c, n) * c + Uc(c, n)) - Ξ,  # foc c
                Un(c, n) - μ * (Unn(c, n) * n + Un(c, n)) + Θ * Ξ,  # foc n
                Θ * n - c - G  # resource constraint
            ])

        # find the root of the FOC
        res = root(FOC, self.zFB)
        if not res.success:
            raise Exception('Could not find LS allocation.')
        z = res.x
        c, n, Ξ = z[:S], z[S:2 * S], z[2 * S:]

        # now compute x
        I = Uc(c, n) * c + Un(c, n) * n
        x = np.linalg.solve(np.eye(S) - self.β * self.π, I)

        return c, n, x, Ξ

    def time0_allocation(self, B_, s_0):
        '''
        Finds the optimal allocation given initial government debt B_ and state s_0
        '''
        model, π, Θ, G, β = self.model, self.π, self.Θ, self.G, self.β
        Uc, Ucc, Un, Unn = model.Uc, model.Ucc, model.Un, model.Unn

        # first order conditions of planner's problem
        def FOC(z):
            μ, c, n, Ξ = z
            xprime = self.time1_allocation(μ)[2]
            return np.hstack([
                Uc(c, n) * (c - B_) + Un(c, n) *
                n + β * π[s_0].dot(xprime),
                Uc(c, n) - μ * (Ucc(c, n) * (c - B_) + Uc(c, n)) - Ξ,
                Un(c, n) - μ * (Unn(c, n) * n + Un(c, n)) + Θ[s_0] * Ξ,
                (Θ * n - c - G)[s_0]
            ])

        # find root
        res = root(FOC, np.array(
            [0., self.cFB[s_0], self.nFB[s_0], self.ΞFB[s_0]]))
        if not res.success:
            raise Exception('Could not find time 0 LS allocation.')

        return res.x

    def time1_value(self, μ):
        '''
        Find the value associated with multiplier μ
        '''
        c, n, x, Ξ = self.time1_allocation(μ)
        U = self.model.U(c, n)
        V = np.linalg.solve(np.eye(self.S) - self.β * self.π, U)
        return c, n, x, V

    def Τ(self, c, n):
        '''
        Computes Τ given c,n
        '''
        model = self.model
        Uc, Un = model.Uc(c, n), model.Un(c, n)

        return 1 + Un / (self.Θ * Uc)

    def simulate(self, B_, s_0, T, sHist=None):
        '''
        Simulates planners policies for T periods
        '''
        model, π, β = self.model, self.π, self.β
        Uc = model.Uc

        if sHist is None:
            sHist = self.mc.simulate(T, s_0)

        cHist, nHist, Bhist, ΤHist, μHist = np.zeros((5, T))
        RHist = np.zeros(T - 1)
        # time0
        μ, cHist[0], nHist[0], _ = self.time0_allocation(B_, s_0)
        ΤHist[0] = self.Τ(cHist[0], nHist[0])[s_0]
        Bhist[0] = B_
        μHist[0] = μ

        # time 1 onward
        for t in range(1, T):
            c, n, x, Ξ = self.time1_allocation(μ)
            Τ = self.Τ(c, n)
            u_c = Uc(c, n)
            s = sHist[t]
            Eu_c = π[sHist[t - 1]].dot(u_c)

            cHist[t], nHist[t], Bhist[t], ΤHist[t] = c[s], n[s], x[s] / \
                u_c[s], Τ[s]

            RHist[t - 1] = Uc(cHist[t - 1], nHist[t - 1]) / (β * Eu_c)
            μHist[t] = μ

        return np.array([cHist, nHist, Bhist, ΤHist, sHist, μHist, RHist])
