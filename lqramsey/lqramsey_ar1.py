import numpy as np
from numpy import array

# == Parameters == #
β = 1 / 1.05
ρ, mg = .7, .35
A = np.identity(2)
A[0, :] = ρ, mg * (1-ρ)
C = np.zeros((2, 1))
C[0, 0] = np.sqrt(1 - ρ**2) * mg / 10
Sg = array((1, 0)).reshape(1, 2)
Sd = array((0, 0)).reshape(1, 2)
Sb = array((0, 2.135)).reshape(1, 2)
Ss = array((0, 0)).reshape(1, 2)

economy = Economy(β=β,
                  Sg=Sg,
                  Sd=Sd,
                  Sb=Sb,
                  Ss=Ss,
                  discrete=False,
                  proc=(A, C))

T = 50
path = compute_paths(T, economy)
gen_fig_1(path)
