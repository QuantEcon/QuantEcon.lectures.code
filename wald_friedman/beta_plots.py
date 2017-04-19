"""
Beta distribution plots for Wald--Friedman lecture.

The Beta distributions are discretized for ease of use in the dynamic
programing code used in the lecture.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sb


def make_distribution_plots(f0, f1):
    """
    This generates the figure that shows the initial versions
    of the distributions and plots their combinations.
    """
    fig, axes = plt.subplots(2, figsize=(10, 8))

    axes[0].set_title("Original Distributions")
    axes[0].plot(f0, lw=2, label=r"$f_0$")
    axes[0].plot(f1, lw=2, label=r"$f_1$")

    axes[1].set_title("Mixtures")
    for p in 0.25, 0.5, 0.75:
        y = p*f0 + (1 - p)*f1
        axes[1].plot(y, lw=2, label=r"$p_k$ = {}".format(p))

    for ax in axes:
        ax.legend(fontsize=14)
        ax.set_xlabel(r"$k$ values", fontsize=14)
        ax.set_ylabel(r"probability of $z_k$", fontsize=14)
        ax.set_ylim(0, 0.07)

    fig.tight_layout()
    return fig


p_m1 = np.linspace(0, 1, 50)
f0 = np.clip(st.beta.pdf(p_m1, a=1, b=1), 1e-8, np.inf)
f0 = f0 / np.sum(f0)
f1 = np.clip(st.beta.pdf(p_m1, a=9, b=9), 1e-8, np.inf)
f1 = f1 / np.sum(f1)

fig = make_distribution_plots(f0, f1)

plt.show()


