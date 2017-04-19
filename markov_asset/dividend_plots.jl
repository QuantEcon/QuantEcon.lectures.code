#=
Plot the dividend process and the state process for the Markov asset pricing
lecture.

=#

using QuantEcon
using Plots
using LaTeXStrings

pyplot()

n = 25
mc = tauchen(n, 0.96, 0.25)  
sim_length = 80

x_series = simulate(mc, sim_length; init=round(Int, n / 2))
lambda_series = exp(x_series)
d_series = cumprod(lambda_series) # assumes d_0 = 1

series = [x_series lambda_series d_series log(d_series)]
labels = [L"$X_t$" L"$g_t$" L"$d_t$" L"$log (d_t)$"]
plot(series, layout=4, labels=labels)

