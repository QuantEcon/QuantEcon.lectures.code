"""
Computes prices in the Lucas asset pricing model.

Default parameterization.
"""

import matplotlib.pyplot as plt
from lucastree import LucasTree, compute_lt_price

tree = LucasTree()
grid = tree.grid  
price_vals = compute_lt_price(tree)

fig, ax = plt.subplots()
ax.plot(grid, price_vals, lw=2, alpha=0.7, label=r'$p^*(y)$')
ax.set_xlim(min(grid), max(grid))

ax.set_xlabel(r'$y$', fontsize=16)
ax.set_ylabel(r'price', fontsize=16)
ax.legend(loc='upper left')

plt.show()
