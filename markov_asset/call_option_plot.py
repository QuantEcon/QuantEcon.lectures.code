"""
Compare call option prices to consol price.

"""

import matplotlib.pyplot as plt
from asset_pricing import *

ap = AssetPriceModel(beta=0.9)
zeta = 1.0
strike_price = 40

x = ap.mc.state_values
p = consol_price(ap, zeta)
w = call_option(ap, zeta, strike_price)

fig, ax = plt.subplots()
ax.plot(x, p, 'b-', lw=2, label='consol price')
ax.plot(x, w, 'g-', lw=2, label='value of call option')
ax.set_xlabel("state")
ax.legend(loc='upper right')
plt.show()
