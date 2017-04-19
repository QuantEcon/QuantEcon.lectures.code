"""

Provides a class that simulates the dynamics of unemployment and employment in
the lake model. 

"""

import numpy as np

class LakeModel:
    r"""
    Solves the lake model and computes dynamics of unemployment stocks and
    rates.
    
    Parameters:
    ------------
    lmda: scalar
        The job finding rate for currently unemployed workers
    alpha: scalar
        The dismissal rate for currently employed workers
    b : scalar
        Entry rate into the labor force
    d : scalar
        Exit rate from the labor force
    
    """
    def __init__(self, lmda=0.283, alpha=0.013, b=0.0124, d=0.00822):
        self._lmda = lmda
        self._alpha = alpha
        self._b = b
        self._d = d

        self.compute_derived_values()

    def compute_derived_values(self):
        # Unpack names to simplify expression
        lmda, alpha, b, d = self._lmda, self._alpha, self._b, self._d

        self._g = b - d
        self._A = np.array([ [(1-d) * (1-alpha), (1-d) * lmda],
                             [(1-d) * alpha + b, (1-lmda) * (1-d) + b]])

        self._A_hat = self._A / (1 + self._g)
        
    @property
    def g(self):
        return self._g

    @property
    def A(self):
        return self._A

    @property
    def A_hat(self):
        return self._A_hat

    @property
    def lmda(self):
        return self._lmda

    @lmda.setter
    def lmda(self, new_value):
        self._lmda = new_value
        self.compute_derived_values()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_value):
        self._alpha = new_value
        self.compute_derived_values()

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, new_value):
        self._b = new_value
        self.compute_derived_values()

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, new_value):
        self._d = new_value
        self.compute_derived_values()
        

    def rate_steady_state(self, tol=1e-6):
        r"""
        Finds the steady state of the system :math:`x_{t+1} = \hat A x_{t}`
        
        Returns
        --------
        xbar : steady state vector of employment and unemployment rates
        """
        x = 0.5 * np.ones(2)
        error = tol + 1
        while error > tol:
            new_x = self.A_hat @ x
            error = np.max(np.abs(new_x - x))
            x = new_x
        return x
        
    def simulate_stock_path(self, X0, T):
        r"""
        Simulates the the sequence of Employment and Unemployent stocks
        
        Parameters
        ------------
        X0 : array 
            Contains initial values (E0, U0)
        T : int
            Number of periods to simulate
        
        Returns
        --------- 
        X : iterator 
            Contains sequence of employment and unemployment stocks
        """

        X = np.atleast_1d(X0) # recast as array just in case
        for t in range(T):
            yield X
            X = self.A @ X
            
    def simulate_rate_path(self, x0, T):
        r"""
        Simulates the the sequence of employment and unemployent rates.
        
        Parameters
        ------------
        x0 : array 
            Contains initial values (e0,u0)
        T : int
            Number of periods to simulate
        
        Returns
        ---------
        x : iterator 
            Contains sequence of employment and unemployment rates

        """
        x = np.atleast_1d(x0) # recast as array just in case
        for t in range(T):
            yield x
            x = self.A_hat @ x
        
