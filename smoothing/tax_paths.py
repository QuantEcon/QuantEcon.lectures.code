import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

np.random.seed(1)
N_simul = 150
cp = ConsumptionProblem()

c_bar, b1, b2 = consumption_complete(cp)
debt_complete = np.asarray([b1, b2])

c_path, debt_path, y_path, s_path = consumption_incomplete(cp, N_simul=N_simul)

fig, ax = plt.subplots(1, 2, figsize = (15, 5))

ax[0].set_title('Tax collection paths', fontsize = 17)
ax[0].plot(np.arange(N_simul), c_path, label = 'incomplete market', lw = 3)
ax[0].plot(np.arange(N_simul), c_bar * np.ones(N_simul), label = 'complete market', lw = 3)
ax[0].plot(np.arange(N_simul), y_path, label = 'govt expenditures', lw = 2, color = sb.color_palette()[3],
           alpha = .6, linestyle = '--')
ax[0].legend(loc = 'best', fontsize = 15)
ax[0].set_xlabel('Periods', fontsize = 13)
ax[0].set_ylim([1.4, 2.1])

ax[1].set_title('Government assets paths', fontsize = 17)
ax[1].plot(np.arange(N_simul), debt_path, label = 'incomplete market', lw = 2)
ax[1].plot(np.arange(N_simul), debt_complete[s_path], label = 'complete market', lw = 2)
ax[1].plot(np.arange(N_simul), y_path, label = 'govt expenditures', lw = 2, color = sb.color_palette()[3],
           alpha = .6, linestyle = '--')
ax[1].legend(loc = 'best', fontsize = 15)
ax[1].axhline(0, color = 'k', lw = 1)
ax[1].set_xlabel('Periods', fontsize = 13)

plt.show()

