"""
Author: Sebastian Graves

Provides a class called ChangModel to solve different 
parameterizations of the Chang (1998) model.
"""

import numpy as np
import quantecon as qe
import time

from scipy.spatial import ConvexHull
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.interpolate import UnivariateSpline
import numpy.polynomial.chebyshev as cheb


class ChangModel:
    """
    Class to solve for the competitive and sustainable sets in the Chang (1998)
    model, for different parameterizations.
    """

    def __init__(self, beta, mbar, h_min, h_max, n_h, n_m, N_g):
        # Record parameters
        self.beta, self.mbar, self.h_min, self.h_max = beta, mbar, h_min, h_max
        self.n_h, self.n_m, self.N_g = n_h, n_m, N_g

        # Create other parameters
        self.m_min = 1e-9
        self.m_max = self.mbar
        self.N_a = self.n_h*self.n_m

        # Utility and production functions
        uc = lambda c: np.log(c)
        uc_p = lambda c: 1/c
        v = lambda m: 1/500 * (mbar * m - 0.5 * m**2)**0.5
        v_p = lambda m: 0.5/500 * (mbar * m - 0.5 * m**2)**(-0.5) * (mbar - m)
        u = lambda h, m: uc(f(h, m)) + v(m)

        def f(h, m):
            x = m * (h - 1)
            f = 180 - (0.4 * x)**2
            return f

        def theta(h, m):
            x = m * (h - 1)
            theta = uc_p(f(h, m)) * (m + x)
            return theta

        # Create set of possible action combinations, A
        A1 = np.linspace(h_min, h_max, n_h).reshape(n_h, 1)
        A2 = np.linspace(self.m_min, self.m_max, n_m).reshape(n_m, 1)
        self.A = np.concatenate((np.kron(np.ones((n_m, 1)), A1),
                                 np.kron(A2, np.ones((n_h, 1)))), axis=1)

        # Pre-compute utility and output vectors
        self.EulerVec = -np.multiply(self.A[:, 1], uc_p(f(self.A[:, 0], self.A[:, 1])) - v_p(self.A[:, 1]))
        self.UVec = u(self.A[:, 0], self.A[:, 1])
        self.ThetaVec = theta(self.A[:, 0], self.A[:, 1])
        self.FVec = f(self.A[:, 0], self.A[:, 1])
        self.BellVec = np.multiply(uc_p(f(self.A[:, 0],
                                   self.A[:, 1])),
                                   np.multiply(self.A[:, 1],
                                   (self.A[:, 0] - 1))) + np.multiply(self.A[:, 1],
                                   v_p(self.A[:, 1]))

        # Find extrema of (w, theta) space for initial guess of equilibrium sets
        P_vec = np.zeros(self.N_a)
        W_vec = np.zeros(self.N_a)
        for i in range(self.N_a):
            P_vec[i] = self.ThetaVec[i]
            W_vec[i] = self.UVec[i]/(1 - beta)

        W_space = np.array([min(W_vec[~np.isinf(W_vec)]),
                            max(W_vec[~np.isinf(W_vec)])])
        P_space = np.array([0, max(P_vec[~np.isinf(W_vec)])])
        self.P_space = P_space

        # Set up hyperplane levels and gradients for iterations
        def SG_H_V(N, W_space, P_space):
            """
            This function  initializes the subgradients, hyperplane levels,
            and extreme points of the value set by choosing an appropriate
            origin and radius. It is based on a similar function in QuantEcon's Games.jl
            """

            # First, create unit circle. Want points placed on [0, 2π]
            inc = 2 * np.pi / N
            degrees = np.arange(0, 2 * np.pi, inc)

            # Points on circle
            H = np.zeros((N, 2))
            for i in range(N):
                x = degrees[i]
                H[i, 0] = np.cos(x)
                H[i, 1] = np.sin(x)

            # Then calculate origin and radius
            o = np.array([np.mean(W_space), np.mean(P_space)])
            r1 = max((max(W_space) - o[0])**2, (o[0] - min(W_space))**2)
            r2 = max((max(P_space) - o[1])**2, (o[1] - min(P_space))**2)
            r = np.sqrt(r1 + r2)

            # Now calculate vertices
            Z = np.zeros((2, N))
            for i in range(N):
                Z[0, i] = o[0] + r*H.T[0, i]
                Z[1, i] = o[1] + r*H.T[1, i]

            # Corresponding hyperplane levels
            C = np.zeros(N)
            for i in range(N):
                C[i] = np.dot(Z[:, i], H[i, :])

            return C, H, Z

        C, self.H, Z = SG_H_V(N_g, W_space, P_space)
        C = C.reshape(N_g, 1)
        self.C0_C, self.C0_S, self.C1_C, self.C1_S = np.copy(C), np.copy(C), np.copy(C), np.copy(C)
        self.Z0_S, self.Z0_C, self.Z1_S, self.Z1_C = np.copy(Z), np.copy(Z), np.copy(Z), np.copy(Z)

        self.W_bnds_S, self.W_bnds_C = (W_space[0], W_space[1]), (W_space[0], W_space[1])
        self.P_bnds_S, self.P_bnds_C = (P_space[0], P_space[1]), (P_space[0], P_space[1])

        # Create dictionaries to save equilibrium set for each iteration
        self.C_dic_S, self.C_dic_C = {}, {}
        self.C_dic_S[0], self.C_dic_C[0] = self.C0_S, self.C0_C

    def solve_worst_spe(self):
        """
        Method to solve for BR(Z). See p.449 of Chang (1998)
        """

        P_vec = np.full(self.N_a, np.nan)
        c = [1, 0]

        # Pre-compute constraints
        Aineq_mbar = np.vstack((self.H, np.array([0, -self.beta])))
        bineq_mbar = np.vstack((self.C0_S, 0))

        Aineq = self.H
        bineq = self.C0_S
        Aeq = [[0, -self.beta]]

        for j in range(self.N_a):
            # Only try if consumption is possible
            if self.FVec[j] > 0:
                # If m = mbar, use inequality constraint
                if self.A[j, 1] == self.mbar:
                    bineq_mbar[-1] = self.EulerVec[j]
                    res = linprog(c, A_ub=Aineq_mbar, b_ub=bineq_mbar, 
                                  bounds=(self.W_bnds_S, self.P_bnds_S))
                else:
                    beq = self.EulerVec[j]
                    res = linprog(c, A_ub=Aineq, b_ub=bineq, A_eq=Aeq, b_eq=beq,
                                  bounds=(self.W_bnds_S, self.P_bnds_S))
                if res.status == 0:
                    P_vec[j] = self.UVec[j] + self.beta * res.x[0]

        # Max over h and min over other variables (see Chang (1998) p.449)
        self.BR_Z = np.nanmax(np.nanmin(P_vec.reshape(self.n_m, self.n_h), 0))

    def solve_subgradient(self):
        """
        Method to solve for E(Z). See p.449 of Chang (1998)
        """

        # Pre-compute constraints
        Aineq_C_mbar = np.vstack((self.H, np.array([0, -self.beta])))
        bineq_C_mbar = np.vstack((self.C0_C, 0))

        Aineq_C = self.H
        bineq_C = self.C0_C
        Aeq_C = [[0, -self.beta]]

        Aineq_S_mbar = np.vstack((np.vstack((self.H, np.array([0, -self.beta]))),
                                  np.array([-self.beta, 0])))
        bineq_S_mbar = np.vstack((self.C0_S, np.zeros((2, 1))))

        Aineq_S = np.vstack((self.H, np.array([-self.beta, 0])))
        bineq_S = np.vstack((self.C0_S, 0))
        Aeq_S = [[0, -self.beta]]

        # Update maximal hyperplane level
        for i in range(self.N_g):
            C_A1A2_C, T_A1A2_C = np.full(self.N_a, -np.inf), np.zeros((self.N_a, 2))
            C_A1A2_S, T_A1A2_S = np.full(self.N_a, -np.inf), np.zeros((self.N_a, 2))

            c = [-self.H[i, 0], -self.H[i, 1]]

            for j in range(self.N_a):
                # Only try if consumption is possible
                if self.FVec[j] > 0:
                
                    # COMPETITIVE EQUILIBRIA
                    # If m = mbar, use inequality constraint
                    if self.A[j, 1] == self.mbar:
                        bineq_C_mbar[-1] = self.EulerVec[j]
                        res = linprog(c, A_ub=Aineq_C_mbar, b_ub=bineq_C_mbar,
                                      bounds=(self.W_bnds_C, self.P_bnds_C))
                    # If m < mbar, use equality constraint
                    else:
                        beq_C = self.EulerVec[j]
                        res = linprog(c, A_ub=Aineq_C, b_ub=bineq_C, A_eq = Aeq_C,
                                      b_eq = beq_C, bounds=(self.W_bnds_C, self.P_bnds_C))
                    if res.status == 0:
                        C_A1A2_C[j] = self.H[i, 0]*(self.UVec[j] + self.beta * res.x[0]) + self.H[i, 1] * self.ThetaVec[j]
                        T_A1A2_C[j] = res.x

                    # SUSTAINABLE EQUILIBRIA
                    # If m = mbar, use inequality constraint
                    if self.A[j, 1] == self.mbar:
                        bineq_S_mbar[-2] = self.EulerVec[j]
                        bineq_S_mbar[-1] = self.UVec[j] - self.BR_Z
                        res = linprog(c, A_ub=Aineq_S_mbar, b_ub=bineq_S_mbar, 
                                      bounds=(self.W_bnds_S, self.P_bnds_S))
                    # If m < mbar, use equality constraint
                    else:
                        bineq_S[-1] = self.UVec[j] - self.BR_Z
                        beq_S = self.EulerVec[j]
                        res = linprog(c, A_ub=Aineq_S, b_ub=bineq_S, A_eq = Aeq_S,
                                      b_eq = beq_S, bounds=(self.W_bnds_S, self.P_bnds_S))
                    if res.status == 0:
                        C_A1A2_S[j] = self.H[i, 0] * (self.UVec[j] + self.beta*res.x[0]) + self.H[i, 1] * self.ThetaVec[j]
                        T_A1A2_S[j] = res.x

            idx_C = np.where(C_A1A2_C == max(C_A1A2_C))[0][0]
            self.Z1_C[:, i] = np.array([self.UVec[idx_C] + self.beta * T_A1A2_C[idx_C, 0],
                                              self.ThetaVec[idx_C]])

            idx_S = np.where(C_A1A2_S == max(C_A1A2_S))[0][0]
            self.Z1_S[:, i] = np.array([self.UVec[idx_S] + self.beta*T_A1A2_S[idx_S, 0],
                                        self.ThetaVec[idx_S]])

        for i in range(self.N_g):
            self.C1_C[i] = np.dot(self.Z1_C[:, i], self.H[i, :])
            self.C1_S[i] = np.dot(self.Z1_S[:, i], self.H[i, :])

    def solve_sustainable(self, tol=1e-5, max_iter=250):
        """
        Method to solve for the competitive and sustainable equilibrium sets.
        """

        t = time.time()
        diff = tol + 1
        iters = 0

        print('### --------------- ###')
        print('Solving Chang Model Using Outer Hyperplane Approximation')
        print('### --------------- ### \n')

        print('Maximum difference when updating hyperplane levels:')

        while diff > tol and iters < max_iter:
            iters = iters + 1
            self.solve_worst_spe()
            self.solve_subgradient()
            diff = max(np.maximum(abs(self.C0_C - self.C1_C),
                       abs(self.C0_S - self.C1_S)))
            print(diff)

            # Update hyperplane levels
            self.C0_C, self.C0_S = np.copy(self.C1_C), np.copy(self.C1_S)

            # Update bounds for w and theta
            Wmin_C, Wmax_C = np.min(self.Z1_C, axis=1)[0], np.max(self.Z1_C, axis=1)[0]
            Pmin_C, Pmax_C = np.min(self.Z1_C, axis=1)[1], np.max(self.Z1_C, axis=1)[1]

            Wmin_S, Wmax_S = np.min(self.Z1_S, axis=1)[0], np.max(self.Z1_S, axis=1)[0]
            Pmin_S, Pmax_S = np.min(self.Z1_S, axis=1)[1], np.max(self.Z1_S, axis=1)[1]

            self.W_bnds_S, self.W_bnds_C = (Wmin_S, Wmax_S), (Wmin_C, Wmax_C)
            self.P_bnds_S, self.P_bnds_C = (Pmin_S, Pmax_S), (Pmin_C, Pmax_C)

            # Save iteration
            self.C_dic_C[iters], self.C_dic_S[iters] = np.copy(self.C1_C), np.copy(self.C1_S)
            self.iters = iters

        elapsed = time.time() - t
        print('Convergence achieved after {} iterations and {} seconds'.format(iters, round(elapsed, 2)))

    def solve_bellman(self, theta_min, theta_max, order, disp=False, tol=1e-7, maxiters=100):
        """
        Continuous Method to solve the Bellman equation in section 25.3
        """
        mbar = self.mbar
        
        # Utility and production functions
        uc = lambda c: np.log(c)
        uc_p = lambda c: 1 / c
        v = lambda m: 1 / 500 * (mbar * m - 0.5 * m**2)**0.5
        v_p = lambda m: 0.5/500 * (mbar*m - 0.5 * m**2)**(-0.5) * (mbar - m)
        u = lambda h, m: uc(f(h, m)) + v(m)
 
        def f(h, m):
            x = m * (h - 1)
            f = 180 - (0.4 * x)**2
            return f
 
        def theta(h, m):
            x = m * (h - 1)
            theta = uc_p(f(h, m)) * (m + x)
            return theta
 
        # Bounds for Maximization
        lb1 = np.array([self.h_min, 0, theta_min])
        ub1 = np.array([self.h_max, self.mbar - 1e-5, theta_max])
        lb2 = np.array([self.h_min, theta_min])
        ub2 = np.array([self.h_max, theta_max])
 
        # Initialize Value Function coefficents
        # Calculate roots of Chebyshev polynomial
        k = np.linspace(order, 1, order)
        roots = np.cos((2 * k - 1) * np.pi / (2 * order))
        # Scale to approximation space
        s = theta_min + (roots - -1) / 2 * (theta_max - theta_min)
        # Create basis matrix
        Phi = cheb.chebvander(roots, order - 1)
        c = np.zeros(Phi.shape[0])
 
        # Function to minimize and constraints
        def P_fun(x):
            scale = -1 + 2 * (x[2] - theta_min)/(theta_max - theta_min)
            P_fun = - (u(x[0], x[1]) + self.beta * np.dot(cheb.chebvander(scale, order - 1), c))
            return P_fun
 
        def P_fun2(x):
            scale = -1 + 2*(x[1] - theta_min)/(theta_max - theta_min)
            P_fun = - (u(x[0],mbar) + self.beta * np.dot(cheb.chebvander(scale, order - 1), c))
            return P_fun
 
        cons1 = ({'type': 'eq',   'fun': lambda x: uc_p(f(x[0], x[1])) * x[1] * (x[0] - 1) + v_p(x[1]) * x[1] + self.beta * x[2] - theta},
                 {'type': 'eq',   'fun': lambda x: uc_p(f(x[0], x[1])) * x[0] * x[1] - theta})
        cons2 = ({'type': 'ineq', 'fun': lambda x: uc_p(f(x[0], mbar)) * mbar * (x[0] - 1) + v_p(mbar) * mbar + self.beta * x[1] - theta},
                 {'type': 'eq',   'fun': lambda x: uc_p(f(x[0], mbar)) * x[0] * mbar - theta})
 
        bnds1 = np.concatenate([lb1.reshape(3, 1), ub1.reshape(3, 1)], axis=1)
        bnds2 = np.concatenate([lb2.reshape(2, 1), ub2.reshape(2, 1)], axis=1)
 
        # Bellman Iterations
        diff = 1
        iters = 1
 
        while diff > tol:
        # 1. Maximization, given value function guess
            P_iter1 = np.zeros(order)
            for i in range(order):
                theta = s[i]
                res = minimize(P_fun,
                               lb1 + (ub1-lb1) / 2,
                               method='SLSQP',
                               bounds=bnds1,
                               constraints=cons1,
                               tol=1e-10)
                if res.success == True:
                    P_iter1[i] = -P_fun(res.x)
                res = minimize(P_fun2,
                               lb2 + (ub2-lb2) / 2,
                               method='SLSQP',
                               bounds=bnds2,
                               constraints=cons2,
                               tol=1e-10)
                if -P_fun2(res.x) > P_iter1[i] and res.success == True:
                    P_iter1[i] = -P_fun2(res.x)

            # 2. Bellman updating of Value Function coefficients
            c1 = np.linalg.solve(Phi, P_iter1)
            # 3. Compute distance and update
            diff = np.linalg.norm(c - c1)
            if bool(disp == True):
                print(diff)
            c = np.copy(c1)
            iters = iters + 1
            if iters > maxiters:
                print('Convergence failed after {} iterations'.format(maxiters))
                break

        self.theta_grid = s
        self.P_iter = P_iter1
        self.Phi = Phi
        self.c = c
        print('Convergence achieved after {} iterations'.format(iters))

        # Check residuals
        theta_grid_fine = np.linspace(theta_min, theta_max, 100)
        resid_grid = np.zeros(100)
        P_grid = np.zeros(100)
        theta_prime_grid = np.zeros(100)
        m_grid = np.zeros(100)
        h_grid = np.zeros(100)
        for i in range(100):
            theta = theta_grid_fine[i]
            res = minimize(P_fun,
                           lb1 + (ub1-lb1) / 2,
                           method='SLSQP',
                           bounds=bnds1,
                           constraints=cons1,
                           tol=1e-10)
            if res.success == True:
                P = -P_fun(res.x)
                P_grid[i] = P
                theta_prime_grid[i] = res.x[2]
                h_grid[i] = res.x[0]
                m_grid[i] = res.x[1]
            res = minimize(P_fun2,
                           lb2 + (ub2-lb2)/2,
                           method='SLSQP',
                           bounds=bnds2,
                           constraints=cons2,
                           tol=1e-10)
            if -P_fun2(res.x) > P and res.success == True:
                P = -P_fun2(res.x)
                P_grid[i] = P
                theta_prime_grid[i] = res.x[1]
                h_grid[i] = res.x[0]
                m_grid[i] = self.mbar
            scale = -1 + 2 * (theta - theta_min)/(theta_max - theta_min)
            resid_grid[i] = np.dot(cheb.chebvander(scale, order-1), c) - P

        self.resid_grid = resid_grid
        self.theta_grid_fine = theta_grid_fine
        self.theta_prime_grid = theta_prime_grid
        self.m_grid = m_grid
        self.h_grid = h_grid
        self.P_grid = P_grid
        self.x_grid = m_grid * (h_grid - 1)

        # Simulate
        theta_series = np.zeros(31)
        m_series = np.zeros(30)
        h_series = np.zeros(30)

        # Find initial theta
        def ValFun(x):
            scale = -1 + 2*(x - theta_min)/(theta_max - theta_min)
            P_fun = np.dot(cheb.chebvander(scale, order - 1), c)
            return -P_fun

        res = minimize(ValFun,
                      (theta_min + theta_max)/2,
                      bounds=[(theta_min, theta_max)])
        theta_series[0] = res.x

        # Simulate
        for i in range(30):
            theta = theta_series[i]
            res = minimize(P_fun,
                           lb1 + (ub1-lb1)/2,
                           method='SLSQP',
                           bounds=bnds1,
                           constraints=cons1,
                           tol=1e-10)
            if res.success == True:
                P = -P_fun(res.x)
                h_series[i] = res.x[0]
                m_series[i] = res.x[1]
                theta_series[i+1] = res.x[2]
            res2 = minimize(P_fun2,
                            lb2 + (ub2-lb2)/2,
                            method='SLSQP',
                            bounds=bnds2,
                            constraints=cons2,
                            tol=1e-10)
            if -P_fun2(res2.x) > P and res2.success == True:
                h_series[i] = res2.x[0]
                m_series[i] = self.mbar
                theta_series[i+1] = res2.x[1]

        self.theta_series = theta_series
        self.m_series = m_series
        self.h_series = h_series
        self.x_series = m_series * (h_series - 1)
