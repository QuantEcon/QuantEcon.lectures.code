#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#
using Plots
pyplot()
using Distributions
using LaTeXStrings

# == Set parameters == #
srand(42)  # reproducible results
n = 250    # Choice of n
k = 10000  # Number of draws of Y_n
dist = Exponential(1./2.)  # Exponential distribution, lambda = 1/2
μ, s = mean(dist), std(dist)

# == Draw underlying RVs. Each row contains a draw of X_1,..,X_n == #
data = rand(dist, k, n)

# == Compute mean of each row, producing k draws of \bar X_n == #
sample_means = mean(data, 2)

# == Generate observations of Y_n == #
Y = sqrt(n) * (sample_means .- μ)

# == Plot == #
xmin, xmax = -3 * s, 3 * s
histogram(Y, nbins=60, alpha=0.5, xlims=(xmin, xmax),
          norm=true, label="")
xgrid = linspace(xmin, xmax, 200)
plot!(xgrid, pdf.(Normal(0.0, s), xgrid), color=:black,
      linewidth=2, label=LaTeXString("\$N(0, \\sigma^2=$(s^2))\$"),
      legendfont=font(12))
