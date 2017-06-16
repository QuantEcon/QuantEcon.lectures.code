"""
Plots reservation wage against gamma

"""
import matplotlib.pyplot as plt

grid_size = 25  
gamma_vals = np.linspace(0.05, 0.95, grid_size)  
w_bar_vals = np.empty_like(gamma_vals)

mcm = McCallModel()

fig, ax = plt.subplots(figsize=(9, 6))

for i, gamma in enumerate(gamma_vals):
    mcm.gamma = gamma
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set_xlabel('job offer rate')
ax.set_ylabel('reservation wage')
ax.set_xlim(gamma_vals.min(), gamma_vals.max())
txt = r'$\bar w$ as a function of $\gamma$'
ax.plot(gamma_vals, w_bar_vals, 'b-', lw=2, alpha=0.7, label=txt)
ax.legend(loc='upper left')
ax.grid()

plt.show()

