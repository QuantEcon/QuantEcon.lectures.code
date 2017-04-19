using QuantEcon
using Plots
pyplot()

f(x) = 2 .* cos.(6x) .+ sin.(14x) .+ 2.5
c_grid = 0:.2:1
n = length(c_grid)

Af = LinInterp(c_grid, f(c_grid))

f_grid = linspace(0, 1, 150)

plot(f_grid, f, color=:blue, linewidth=2, alpha=0.8, label="true function")
plot!(f_grid, Af.(f_grid), color=:green, linewidth=2, alpha=0.8,
      label="linear approximation", legend=:top, grid=false)
N = repmat(c_grid, 1, 2)'
heights = [zeros(1,n); f(c_grid)']
plot!(N, heights, color=:black, linestyle=:dash, alpha=0.5, label="")
