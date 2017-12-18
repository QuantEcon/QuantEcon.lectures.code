using QuantEcon
using Plots
using LaTeXStrings

pyplot()

n = 25  # size of state space
β = 0.9
mc = tauchen(n, 0.96, 0.02)

K = mc.p .* exp.(mc.state_values)'

I = eye(n)
v = (I - β * K) \  (β * K * ones(n, 1))

plot(mc.state_values,
    v,
    lw=2,
    ylabel="price-dividend ratio",
    xlabel="state",
    alpha=0.7,
    label=L"$v$")
