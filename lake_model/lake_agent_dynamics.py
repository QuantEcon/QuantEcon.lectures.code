from quantecon import MarkovChain

lm = LakeModel(d=0, b=0)
T = 5000  # Simulation length

alpha, lmda = lm.alpha, lm.lmda

P = [[1 - lmda,   lmda],
     [alpha, 1 - alpha]]

mc = MarkovChain(P)

xbar = lm.rate_steady_state()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
s_path = mc.simulate(T, init=1)
s_bar_e = s_path.cumsum() / range(1, T+1)
s_bar_u = 1 - s_bar_e

axes[0].plot(s_bar_u, '-b', lw=2, alpha=0.5)
axes[0].hlines(xbar[0], 0, T, 'r', '--')
axes[0].set_title('Percent of time unemployed')

axes[1].plot(s_bar_e, '-b', lw=2, alpha=0.5)
axes[1].hlines(xbar[1], 0, T, 'r', '--')
axes[1].set_title('Percent of time employed')

plt.tight_layout()
plt.show()