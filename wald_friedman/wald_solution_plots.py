import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
import seaborn as sb
from wald_class import *


c = 1.25
L0 = 25
L1 = 25
a0, b0 = 2.5, 2.0
a1, b1 = 2.0, 2.5
m = 25

f0 = np.clip(st.beta.pdf(np.linspace(0, 1, m), a=a0, b=b0), 1e-6, np.inf)
f0 = f0 / np.sum(f0)
f1 = np.clip(st.beta.pdf(np.linspace(0, 1, m), a=a1, b=b1), 1e-6, np.inf)
f1 = f1 / np.sum(f1)  # Make sure sums to 1

# Create an instance of our WaldFriedman class
wf = WaldFriedman(c, L0, L1, f0, f1, m=m)
# Solve using qe's `compute_fixed_point` function
J = qe.compute_fixed_point(wf.bellman_operator, np.zeros(m),
                           error_tol=1e-7, verbose=False,
                           print_skip=10, max_iter=500)
lb, ub = wf.find_cutoff_rule(J)

# Get draws
ndraws = 500
cdist, tdist = wf.stopping_dist(ndraws=ndraws)

fig, ax = plt.subplots(2, 2, figsize=(12, 9))

ax[0, 0].plot(f0, label=r"$f_0$")
ax[0, 0].plot(f1, label=r"$f_1$")
ax[0, 0].set_ylabel(r"probability of $z_k$", size=14)
ax[0, 0].set_xlabel(r"$k$", size=14)
ax[0, 0].set_title("Distributions", size=14)
ax[0, 0].legend(fontsize=14)

ax[0, 1].plot(wf.pgrid, J)
ax[0, 1].annotate(r"$\beta$", xy=(lb+0.025, 0.5), size=14)
ax[0, 1].annotate(r"$\alpha$", xy=(ub+0.025, 0.5), size=14)
ax[0, 1].vlines(lb, 0.0, wf.payoff_choose_f1(lb), linestyle="--")
ax[0, 1].vlines(ub, 0.0, wf.payoff_choose_f0(ub), linestyle="--")
ax[0, 1].set_ylim(0, 0.5*max(L0, L1))
ax[0, 1].set_ylabel("cost", size=14)
ax[0, 1].set_xlabel(r"$p_k$", size=14)
ax[0, 1].set_title(r"Value function $J$", size=14)

# Histogram the stopping times
ax[1, 0].hist(tdist, bins=np.max(tdist))
ax[1, 0].set_title("Stopping times over {} replications".format(ndraws), size=14)
ax[1, 0].set_xlabel("time", size=14)
ax[1, 0].set_ylabel("number of stops", size=14)
ax[1, 0].annotate("mean = {}".format(np.mean(tdist)), 
        xy=(max(tdist)/2, 
            max(np.histogram(tdist, bins=max(tdist))[0])/2), 
        size=16)

ax[1, 1].hist(cdist, bins=2)
ax[1, 1].set_title("Correct decisions over {} replications".format(ndraws), size=14)
ax[1, 1].annotate("% correct = {}".format(np.mean(cdist)), 
        xy=(0.05, ndraws/2), size=16)

fig.tight_layout()
plt.show()

