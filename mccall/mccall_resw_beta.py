grid_size = 25  
beta_vals = np.linspace(0.8, 0.99, grid_size)  
w_bar_vals = np.empty_like(beta_vals)

mcm = McCallModel()

fig, ax = plt.subplots()

for i, beta in enumerate(beta_vals):
    mcm.beta = beta
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set_xlabel('discount factor')
ax.set_ylabel('reservation wage')
ax.set_xlim(beta_vals.min(), beta_vals.max())
txt = r'$\bar w$ as a function of $\beta$'
ax.plot(beta_vals, w_bar_vals, 'b-', lw=2, alpha=0.7, label=txt)
ax.legend(loc='upper left')
ax.grid()

plt.show()
