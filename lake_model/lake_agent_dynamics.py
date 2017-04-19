"""
Agent dynamics the a lake model.

"""

import numpy as np
import matplotlib.pyplot as plt
from lake_model import LakeModel
from quantecon import MarkovChain
import matplotlib
matplotlib.style.use('ggplot')

lm = LakeModel(d=0, b=0)
T = 5000  # Simulation length

alpha, lmda = lm.alpha, lm.lmda

P = [[1 - lmda, lmda],
     [alpha, 1 - alpha]]

mc = MarkovChain(P)

xbar = lm.rate_steady_state()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
s_path = mc.simulate(T, init=1)
s_bar_e = s_path.cumsum() / range(1, T+1)
s_bar_u = 1 - s_bar_e

ax = axes[0]
ax.plot(s_bar_u, '-b', lw=2, alpha=0.5)
ax.hlines(xbar[1], 0, T, 'r', '--')
ax.set_title(r'Percent of time unemployed')

ax = axes[1]
ax.plot(s_bar_e, '-b', lw=2, alpha=0.5)
ax.hlines(xbar[0], 0, T, 'r', '--')
ax.set_title(r'Percent of time employed')

plt.tight_layout()
plt.show()


