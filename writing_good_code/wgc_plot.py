import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def plot_path(ax, alphas, s_vals, deltas, series_length=50):
    """
    Add a time series plot to the axes ax for all given parameters.
    """
    k = np.empty(series_length)
    label = "$\\alpha = {},\; s = {},\; \\delta = {}$"

    for (alpha, s, delta) in product(alphas, s_vals, deltas):
        k[0] = 1
        for t in range(series_length-1):
            k[t+1] = s * k[t]**alpha + (1 - delta) * k[t]
        ax.plot(k, 'o-', label=label.format(alpha, s, delta))

    ax.grid(lw=0.2)
    ax.set_xlabel('time')
    ax.set_ylabel('capital')
    ax.set_ylim(0, 18)
    ax.legend(loc='upper left', frameon=True, fontsize=14)

fig, axes = plt.subplots(3, 1, figsize=(9, 15))

# Parameters (alphas, s_vals, deltas)
set_one = ([0.25, 0.33, 0.45], [0.4], [0.1])
set_two = ([0.33], [0.3, 0.4, 0.5], [0.1])
set_three = ([0.33], [0.4], [0.05, 0.1, 0.15])

for (ax, params) in zip(axes, (set_one, set_two, set_three)):
    alphas, s_vals, deltas = params
    plot_path(ax, alphas, s_vals, deltas)

plt.show()
