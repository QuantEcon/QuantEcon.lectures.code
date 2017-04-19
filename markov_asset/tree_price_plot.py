"""
Plots of Lucas tree price for different gamma.

"""

import matplotlib.pyplot as plt
from asset_pricing import *

gammas = [1.2, 1.4, 1.6, 1.8, 2.0]
ap = AssetPriceModel()
states = ap.mc.state_values

fig, ax = plt.subplots()

for gamma in gammas:
    ap.gamma = gamma
    v = tree_price(ap)
    label = r"$\gamma = {}$".format(gamma)
    ax.plot(states, v,  lw=2, alpha=0.6, label=label)

ax.set_title('Price-divdend ratio as a function of the state')
ax.set_ylabel("price-dividend ratio")
ax.set_xlabel("state")
ax.legend(loc='upper right')
plt.show()
