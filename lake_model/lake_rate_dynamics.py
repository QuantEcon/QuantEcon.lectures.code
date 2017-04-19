"""
Stock dynamics the a lake model.
"""

import numpy as np
import matplotlib.pyplot as plt
from lake_model import LakeModel
import matplotlib
matplotlib.style.use('ggplot')

lm = LakeModel()
e_0 = 0.92     # Initial employment rate
u_0 = 1 - e_0  # Initial unemployment rate
T = 50         # Simulation length

xbar = lm.rate_steady_state()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
x_0 = (e_0, u_0)
x_path = np.vstack(lm.simulate_rate_path(x_0, T))

ax = axes[0]
ax.plot(x_path[:,0], '-b', lw=2, alpha=0.5)
ax.hlines(xbar[0], 0, T, 'r', '--')
ax.set_title(r'Employment rate')

ax = axes[1]
ax.plot(x_path[:,1], '-b', lw=2, alpha=0.5)
ax.hlines(xbar[1], 0, T, 'r', '--')
ax.set_title(r'Unemployment rate')

plt.tight_layout()
plt.show()
