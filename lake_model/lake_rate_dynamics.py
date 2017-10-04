lm = LakeModel()
e_0 = 0.92     # Initial employment rate
u_0 = 1 - e_0  # Initial unemployment rate
T = 50         # Simulation length

xbar = lm.rate_steady_state()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
x_0 = (u_0, e_0)
x_path = np.vstack(lm.simulate_rate_path(x_0, T))

axes[0].plot(x_path[:,0], '-b', lw=2, alpha=0.5)
axes[0].hlines(xbar[0], 0, T, 'r', '--')
axes[0].set_title('Unemployment rate')

axes[1].plot(x_path[:,1], '-b', lw=2, alpha=0.5)
axes[1].hlines(xbar[1], 0, T, 'r', '--')
axes[1].set_title('Employment rate')

plt.tight_layout()
plt.show()