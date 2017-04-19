"""
Classical discrete time LQ optimal control and filtering problems. The
control problems we consider take the form

    \max \sum_{t = 0}^N \beta^t \{a_t y_t - h y^2_t / 2 - [d(L)y_t]^2 \} 
    
subject to h > 0, 0 < \beta < 1 and 

 * y_{-1},y_{-2}, \dots, y_{-m} (initial conditions)

 * d(L) = d_0 + d_1L + d_2L^2 + \dots + d_mL^m

The sequence {y_t} is scalar

Authors: Balint Skoze, Tom Sargent, John Stachurski

"""

import numpy as np
import scipy.stats as spst
import scipy.linalg as la

class LQFilter:
    
    def __init__(self, d, h, y_m, r=None, h_eps=None, beta=None):
        """
        
        Parameters
        ----------
            d : list or numpy.array (1-D or a 2-D column vector)
                    The order of the coefficients: [d_0, d_1, ..., d_m]
            h : scalar
                    Parameter of the objective function (corresponding to the
                    quadratic term)
            y_m : list or numpy.array (1-D or a 2-D column vector)
                    Initial conditions for y
            r : list or numpy.array (1-D or a 2-D column vector)
                    The order of the coefficients: [r_0, r_1, ..., r_k] 
                    (optional, if not defined -> deterministic problem)
            beta : scalar
                    Discount factor (optional, default value is one)
            
        """
        
        self.h = h
        self.d = np.asarray(d)
        self.m = self.d.shape[0] - 1
        
        self.y_m = np.asarray(y_m)
    
        if self.m == self.y_m.shape[0]:
            self.y_m = self.y_m.reshape(self.m, 1)
        else:
            raise ValueError("y_m must be of length m = {:d}".format(self.m))
        
        #---------------------------------------------
        # Define the coefficients of phi up front
        #---------------------------------------------
        phi = np.zeros(2 * self.m + 1)
        for i in range(- self.m, self.m + 1):
            phi[self.m - i] = np.sum(np.diag(self.d.reshape(self.m + 1, 1) @ \
                                                  self.d.reshape(1, self.m + 1), k = -i))
        phi[self.m] = phi[self.m] + self.h
        self.phi = phi

        #-----------------------------------------------------
        # If r is given calculate the vector phi_r
        #-----------------------------------------------------
        if r is None:
            pass
        else:
            self.r = np.asarray(r)
            self.k = self.r.shape[0] - 1
            phi_r = np.zeros(2 * self.k + 1)
            for i in range(- self.k, self.k + 1):
                phi_r[self.k - i] = np.sum(np.diag(self.r.reshape(self.k + 1, 1) @ \
                                                   self.r.reshape(1, self.k + 1), k = -i))
            if h_eps is None:
                self.phi_r = phi_r
            else:
                phi_r[self.k] = phi_r[self.k] + h_eps
                self.phi_r = phi_r        
        
        #-----------------------------------------------------
        # If beta is given, define the transformed variables
        #-----------------------------------------------------
        if beta is None:
            self.beta = 1
        else:
            self.beta = beta
            self.d = self.beta**(np.arange(self.m + 1)/2) * self.d
            self.y_m = self.y_m * (self.beta**(- np.arange(1, self.m + 1)/2)).reshape(self.m, 1)
        
        
    def construct_W_and_Wm(self, N):
        """
        This constructs the matrices W and W_m for a given number of periods N
        """
        
        m = self.m
        d = self.d
        
        W = np.zeros((N + 1, N + 1))
        W_m = np.zeros((N + 1, m))

        #---------------------------------------
        # Terminal conditions
        #---------------------------------------
        
        D_m1 = np.zeros((m + 1, m + 1))
        M = np.zeros((m + 1, m))

        # (1) Constuct the D_{m+1} matrix using the formula

        for j in range(m + 1):
            for k in range(j, m + 1):
                D_m1[j, k] = d[:j + 1] @ d[k - j : k + 1] 
        
        # Make the matrix symmetric 
        D_m1 = D_m1 + D_m1.T - np.diag(np.diag(D_m1))
    
        # (2) Construct the M matrix using the entries of D_m1
        
        for j in range(m):
            for i in range(j + 1, m + 1):
                M[i, j] = D_m1[i - j - 1, m]
 
        #----------------------------------------------
        # Euler equations for t = 0, 1, ..., N-(m+1)  
        #----------------------------------------------
        phi = self.phi
        
        W[:(m + 1), :(m + 1)] = D_m1 + self.h * np.eye(m + 1)
        W[:(m + 1), (m + 1):(2 * m + 1)] = M

        for i, row in enumerate(np.arange(m + 1, N + 1 - m)):
            W[row, (i + 1):(2 * m + 2 + i)] = phi
    
        for i in range(1, m + 1):
            W[N - m + i, -(2 * m + 1 - i):] = phi[:-i]

        for i in range(m):
            W_m[N - i, :(m - i)] = phi[(m + 1 + i):]
        
        return W, W_m
        
        

    def roots_of_characteristic(self):
        """
        This function calculates z_0 and the 2m roots of the characteristic equation 
        associated with the Euler equation (1.7)
        
        Note:
        ------
        numpy.poly1d(roots, True) defines a polynomial using its roots that can be
        evaluated at any point. If x_1, x_2, ... , x_m are the roots then 
            p(x) = (x - x_1)(x - x_2)...(x - x_m)
        
        """
        m = self.m
        phi = self.phi
        
        # Calculate the roots of the 2m-polynomial
        roots = np.roots(phi)
        # sort the roots according to their length (in descending order) 
        roots_sorted = roots[np.argsort(abs(roots))[::-1]]    

        z_0 = phi.sum() / np.poly1d(roots, True)(1)
        z_1_to_m = roots_sorted[:m]     # we need only those outside the unit circle
        
        lambdas = 1 / z_1_to_m

        return z_1_to_m, z_0, lambdas

    
    def coeffs_of_c(self):
        '''
        This function computes the coefficients {c_j, j = 0, 1, ..., m} for
                c(z) = sum_{j = 0}^{m} c_j z^j
        
        Based on the expression (1.9). The order is
            c_coeffs = [c_0, c_1, ..., c_{m-1}, c_m]
        '''
        z_1_to_m, z_0 = self.roots_of_characteristic()[:2]
        
        c_0 = (z_0 * np.prod(z_1_to_m).real * (- 1)**self.m)**(.5)
        c_coeffs = np.poly1d(z_1_to_m, True).c * z_0 / c_0
        
        return c_coeffs[::-1]
        
        
    def solution(self):
        """
        This function calculates {lambda_j, j=1,...,m} and {A_j, j=1,...,m}
        of the expression (1.15)
        """
        lambdas = self.roots_of_characteristic()[2]
        c_0 = self.coeffs_of_c()[-1]
        
        A = np.zeros(self.m, dtype = complex)
        for j in range(self.m):
            denom = 1 - lambdas/lambdas[j]
            A[j] = c_0**(-2) / np.prod(denom[np.arange(self.m) != j])
        
        return lambdas, A
    
    
    def construct_V(self, N):
        '''
        This function constructs the covariance matrix for x^N (see section 6)
        for a given period N
        '''
        V = np.zeros((N, N))
        phi_r = self.phi_r
        
        for i in range(N):
            for j in range(N):
                if abs(i-j) <= self.k:
                    V[i, j] = phi_r[self.k + abs(i-j)]
                    
        return V
    
    
    def simulate_a(self, N):
        """
        Assuming that the u's are normal, this method draws a random path 
        for x^N  
        """
        V = self.construct_V(N + 1)
        d = spst.multivariate_normal(np.zeros(N + 1), V)
        
        return d.rvs()
    
    
    def predict(self, a_hist, t):
        """
        This function implements the prediction formula discussed is section 6 (1.59)
        It takes a realization for a^N, and the period in which the prediciton is formed
        
        Output:  E[abar | a_t, a_{t-1}, ..., a_1, a_0]
        """
        
        N = np.asarray(a_hist).shape[0] - 1        
        a_hist = np.asarray(a_hist).reshape(N + 1, 1)
        V = self.construct_V(N + 1)
        
        aux_matrix = np.zeros((N + 1, N + 1))
        aux_matrix[:(t + 1), :(t + 1)] = np.eye(t + 1)
        L = la.cholesky(V).T
        Ea_hist = la.inv(L) @ aux_matrix @ L @ a_hist
        
        return Ea_hist
    
    
    def optimal_y(self, a_hist, t = None):
        """
        - if t is NOT given it takes a_hist (list or numpy.array) as a deterministic a_t
        - if t is given, it solves the combined control prediction problem (section 7)
          (by default, t == None -> deterministic)
            
        for a given sequence of a_t (either determinstic or a particular realization), 
        it calculates the optimal y_t sequence using the method of the lecture

        Note: 
        ------
        scipy.linalg.lu normalizes L, U so that L has unit diagonal elements
        To make things cosistent with the lecture, we need an auxiliary diagonal 
        matrix D which renormalizes L and U
        """
        
        N = np.asarray(a_hist).shape[0] - 1        
        W, W_m = self.construct_W_and_Wm(N)
                 
        L, U = la.lu(W, permute_l = True)
        D = np.diag(1/np.diag(U)) 
        U = D @ U
        L = L @ np.diag(1/np.diag(D))

        J = np.fliplr(np.eye(N + 1))

        if t is None:   # if the problem is deterministic
            
            a_hist = J @ np.asarray(a_hist).reshape(N + 1, 1)

            #--------------------------------------------
            # Transform the a sequence if beta is given
            #--------------------------------------------
            if self.beta != 1:
                a_hist =  a_hist * (self.beta**(np.arange(N + 1) / 2))[::-1].reshape(N + 1, 1) 
            
            a_bar = a_hist - W_m @ self.y_m           # a_bar from the lecutre
            Uy = np.linalg.solve(L, a_bar)            # U @ y_bar = L^{-1}a_bar from the lecture
            y_bar = np.linalg.solve(U, Uy)            # y_bar = U^{-1}L^{-1}a_bar
        
            # Reverse the order of y_bar with the matrix J
            J = np.fliplr(np.eye(N + self.m + 1))
            y_hist = J @ np.vstack([y_bar, self.y_m])     # y_hist : concatenated y_m and y_bar
        
            #--------------------------------------------
            # Transform the optimal sequence back if beta is given
            #--------------------------------------------
            if self.beta != 1:
                y_hist * (self.beta**(- np.arange(-self.m, N + 1)/2)).reshape(N + 1 + self.m, 1) 
        
        
            return y_hist, L, U, y_bar
        
        else:           # if the problem is stochastic and we look at it 
            
            Ea_hist = self.predict(a_hist, t).reshape(N + 1, 1)
            Ea_hist = J @ Ea_hist
                        
            a_bar = Ea_hist - W_m @ self.y_m           # a_bar from the lecutre
            Uy = np.linalg.solve(L, a_bar)            # U @ y_bar = L^{-1}a_bar from the lecture
            y_bar = np.linalg.solve(U, Uy)            # y_bar = U^{-1}L^{-1}a_bar
        
            # Reverse the order of y_bar with the matrix J
            J = np.fliplr(np.eye(N + self.m + 1))
            y_hist = J @ np.vstack([y_bar, self.y_m])     # y_hist : concatenated y_m and y_bar
                
            return y_hist, L, U, y_bar
    
