
def plot_timeseries(n1_0, n2_0, s1=0.5, theta=2.5, delta=0.7, rho=0.2, ax=None):
    """
    Plot a single time series with initial conditions
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Create the MSG Model and simulate with initial conditions
    model = MSGSync(s1, theta, delta, rho)
    n1, n2 = model.simulate_n(n1_0, n2_0, 25)

    ax.plot(np.arange(25), n1, label=r"$n_1$", lw=2)
    ax.plot(np.arange(25), n2, label=r"$n_2$", lw=2)

    ax.legend()
    ax.set_ylim(0.15, 0.8)

    return ax


# Create figure
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

plot_timeseries(0.15, 0.35, ax=ax[0])
plot_timeseries(0.4, 0.3, ax=ax[1])

ax[0].set_title("Not Synchronized")
ax[1].set_title("Synchronized")

fig.tight_layout()

plt.show()
