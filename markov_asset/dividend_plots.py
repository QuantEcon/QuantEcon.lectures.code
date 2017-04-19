"""
Plot the dividend process and the state process for the Markov asset pricing
lecture.

"""
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe

mc = qe.tauchen(0.96, 0.25, n=25)  
sim_length = 80

x_series = mc.simulate(sim_length, init=np.median(mc.state_values))
lambda_series = np.exp(x_series)
d_series = np.cumprod(lambda_series) # assumes d_0 = 1

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(x_series, 'b-', lw=2, label=r'$X_t$')
axes[0, 1].plot(lambda_series, 'b-', lw=2, label=r'$g_t$')
axes[1, 0].plot(d_series, 'b-', lw=2, label=r'$d_t$')
axes[1, 1].plot(np.log(d_series), 'b-', lw=2, label=r'$\log \, d_t$')
for ax in axes.flatten():
    ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.show()
