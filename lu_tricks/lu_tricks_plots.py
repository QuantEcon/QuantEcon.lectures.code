import numpy as np
import matplotlib.pyplot as plt

# == Set seed and generate a_t sequence == #
np.random.seed(123)
n = 100
a_seq = np.sin(np.linspace(0, 5*np.pi, n)) + 2 + 0.1 * np.random.randn(n)

# == Model parameters == #
γ = 0.8
m = 1 
d = γ * np.asarray([1, -1])
h = 1.0 

# == Initial conditions == #
y_m = np.asarray([2.0]).reshape(m, 1)

testlq = LQFilter(d, h, y_m)
y_hist, L, U, y = testlq.optimal_y(a_seq)
y = y[::-1]  # reverse y

# == Plot simulation results == #
fig, ax = plt.subplots(figsize=(10, 6.1))
ax.set_xlabel('Time')

# == Some fancy plotting stuff -- simplify if you prefer == #
bbox = (0., 1.01, 1., .101)
legend_args = {'bbox_to_anchor' : bbox, 'loc' : 3, 'mode' : 'expand', 'fontsize': 16}
p_args = {'lw' : 2, 'alpha' : 0.6}

time = range(len(y))
ax.set_xlim(0, max(time))
ax.plot(time, a_seq / h, 'k-o', ms=4, lw=2, alpha=0.6, label=r'$a_t$')
ax.plot(time, y, 'b-o', ms=4, lw=2, alpha=0.6, label=r'$y_t$')
ax.legend(ncol=2, **legend_args)
ax.grid()
s = r'dynamics with $\gamma = {}$'.format(γ)
plt.show()
