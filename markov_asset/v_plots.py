import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from scipy.linalg import solve, eigvals

n = 25  # size of state space
beta = 0.9
mc = qe.tauchen(0.96, 0.02, n=n)  

K = mc.P * np.exp(mc.state_values)

warning_message = "Spectral radius condition fails"
assert np.max(np.abs(eigvals(K))) < 1 / beta,  warning_message

I = np.identity(n)
v = solve(I - beta * K, beta * K @ np.ones(n))

fig, ax = plt.subplots()
ax.plot(mc.state_values, v, 'g-o', lw=2, alpha=0.7, label=r'$v$')
ax.set_ylabel("price-dividend ratio")
ax.set_xlabel("state")
ax.legend(loc='upper left')
plt.show()
