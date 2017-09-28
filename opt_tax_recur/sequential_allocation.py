import numpy as np
from scipy.optimize import root
from quantecon import MarkovChain


class SequentialAllocation:


    '''
    Class that takes CESutility or BGPutility object as input returns
    planner's allocation as a function of the multiplier on the
    implementability constraint mu.
    '''

    def __init__(self, model):

        # Initialize from model object attributes
        self.beta, self.pi, self.G = model.beta, model.pi, model.G
        self.mc, self.Theta = MarkovChain(self.pi), model.Theta
        self.S = len(model.pi)  # Number of states
        self.model = model

        # Find the first best allocation
        self.find_first_best()

    def find_first_best(self):
        '''
        Find the first best allocation
        '''
        model = self.model
        S, Theta, G = self.S, self.Theta, self.G
        Uc, Un = model.Uc, model.Un

        def res(z):
            c = z[:S]
            n = z[S:]
            return np.hstack([Theta * Uc(c, n) + Un(c, n), Theta * n - c - G])

        res = root(res, 0.5 * np.ones(2 * S))

        if not res.success:
            raise Exception('Could not find first best')

        self.cFB = res.x[:S]
        self.nFB = res.x[S:]

        # Multiplier on the resource constraint
        self.XiFB = Uc(self.cFB, self.nFB)
        self.zFB = np.hstack([self.cFB, self.nFB, self.XiFB])

    def time1_allocation(self, mu):
        '''
        Computes optimal allocation for time t\geq 1 for a given \mu
        '''
        model = self.model
        S, Theta, G = self.S, self.Theta, self.G
        Uc, Ucc, Un, Unn = model.Uc, model.Ucc, model.Un, model.Unn

        def FOC(z):
            c = z[:S]
            n = z[S:2 * S]
            Xi = z[2 * S:]
            return np.hstack([Uc(c, n) - mu * (Ucc(c, n) * c + Uc(c, n)) - Xi,          # FOC of c
                              Un(c, n) - mu * (Unn(c, n) * n + Un(c, n)) + \
                              Theta * Xi,  # FOC of n
                              Theta * n - c - G])

        # Find the root of the first order condition
        res = root(FOC, self.zFB)
        if not res.success:
            raise Exception('Could not find LS allocation.')
        z = res.x
        c, n, Xi = z[:S], z[S:2 * S], z[2 * S:]

        # Compute x
        I = Uc(c, n) * c + Un(c, n) * n
        x = np.linalg.solve(np.eye(S) - self.beta * self.pi, I)

        return c, n, x, Xi

    def time0_allocation(self, B_, s_0):
        '''
        Finds the optimal allocation given initial government debt B_ and state s_0
        '''
        model, pi, Theta, G, beta = self.model, self.pi, self.Theta, self.G, self.beta
        Uc, Ucc, Un, Unn = model.Uc, model.Ucc, model.Un, model.Unn

        # First order conditions of planner's problem
        def FOC(z):
            mu, c, n, Xi = z
            xprime = self.time1_allocation(mu)[2]
            return np.hstack([Uc(c, n) * (c - B_) + Un(c, n) * n + beta * pi[s_0] @ xprime,
                              Uc(c, n) - mu * (Ucc(c, n) *
                                               (c - B_) + Uc(c, n)) - Xi,
                              Un(c, n) - mu * (Unn(c, n) * n +
                                               Un(c, n)) + Theta[s_0] * Xi,
                              (Theta * n - c - G)[s_0]])

        # Find root
        res = root(FOC, np.array(
            [0, self.cFB[s_0], self.nFB[s_0], self.XiFB[s_0]]))
        if not res.success:
            raise Exception('Could not find time 0 LS allocation.')

        return res.x

    def time1_value(self, mu):
        '''
        Find the value associated with multiplier mu
        '''
        c, n, x, Xi = self.time1_allocation(mu)
        U = self.model.U(c, n)
        V = np.linalg.solve(np.eye(self.S) - self.beta * self.pi, U)
        return c, n, x, V

    def Tau(self, c, n):
        '''
        Computes Tau given c, n
        '''
        model = self.model
        Uc, Un = model.Uc(c, n), model.Un(c,  n)

        return 1 + Un / (self.Theta * Uc)

    def simulate(self, B_, s_0, T, sHist=None):
        '''
        Simulates planners policies for T periods
        '''
        model, pi, beta = self.model, self.pi, self.beta
        Uc = model.Uc

        if sHist is None:
            sHist = self.mc.simulate(T, s_0)

        cHist, nHist, Bhist, TauHist, muHist = np.zeros((5, T))
        RHist = np.zeros(T - 1)

        # Time 0
        mu, cHist[0], nHist[0], _ = self.time0_allocation(B_, s_0)
        TauHist[0] = self.Tau(cHist[0], nHist[0])[s_0]
        Bhist[0] = B_
        muHist[0] = mu

        # Time 1 onward
        for t in range(1, T):
            c, n, x, Xi = self.time1_allocation(mu)
            Tau = self.Tau(c, n)
            u_c = Uc(c, n)
            s = sHist[t]
            Eu_c = pi[sHist[t - 1]] @ u_c
            cHist[t], nHist[t], Bhist[t], TauHist[t] = c[s], n[s], x[s] / \
                u_c[s], Tau[s]
            RHist[t - 1] = Uc(cHist[t - 1], nHist[t - 1]) / (beta * Eu_c)
            muHist[t] = mu

        return np.array([cHist, nHist, Bhist, TauHist, sHist, muHist, RHist])
