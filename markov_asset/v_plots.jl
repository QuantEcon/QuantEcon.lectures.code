
#=

Plot the price-dividend ratio in the risk neutral case, for the Markov asset pricing
lecture.

=#
using QuantEcon
using Plots
using LaTeXStrings

pyplot()

n = 25  # size of state space
beta = 0.9
mc = tauchen(n, 0.96, 0.02)  

K = mc.p .* reshape(exp(mc.state_values), 1, n)

I = eye(n)
v = (I - beta * K) \  (beta * K * ones(n, 1))

plot(mc.state_values, 
    v, 
    lw=2, 
    ylabel="price-dividend ratio", 
    xlabel="state", 
    alpha=0.7, 
    label=L"$v$")
