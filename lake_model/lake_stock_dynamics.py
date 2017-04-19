"""
Stock dynamics the a lake model.
"""

import numpy as np
import matplotlib.pyplot as plt
from lake_model import LakeModel
import matplotlib
matplotlib.style.use('ggplot')

lm = LakeModel()
N_0 = 150      # Population
e_0 = 0.92     # Initial employment rate
u_0 = 1 - e_0  # Initial unemployment rate
T = 50         # Simulation length

E_0 = e_0 * N_0
U_0 = u_0 * N_0

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
X_0 = (E_0, U_0)
X_path = np.vstack(lm.simulate_stock_path(X_0, T))

ax = axes[0]
ax.plot(X_path[:,0], '-b', lw=2, alpha=0.7)
ax.set_title(r'Employment')

ax = axes[1]
ax.plot(X_path[:,1], '-b', lw=2, alpha=0.7)
ax.set_title(r'Unemployment')

ax = axes[2]
ax.plot(X_path.sum(1), '-b', lw=2, alpha=0.7)
ax.set_title(r'Labor Force')

plt.tight_layout()
plt.show()
