#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#

using Plots
pyplot()
using Distributions
using LaTeXStrings

srand(12)  # reproducible results
n = 200
dist = Cauchy()
data = rand(dist, n)

function plot_draws()
    t = "$n observations from the Cauchy distribution"
    N = repmat(linspace(1, n, n), 1, 2)'
    heights = [zeros(1,n); data']
    plot(1:n, data, color=:blue, markershape=:circle,
         alpha=0.5, title=t, legend=:none, linewidth=0)
    plot!(N, heights, linewidth=0.5, color=:blue)
end


function plot_means()
    # == Compute sample mean at each n == #
    sample_mean = Array{Float64}(n)
    for i=1:n
        sample_mean[i] = mean(data[1:i])
    end

    # == Plot == #
    plot(1:n, sample_mean, color=:red,
         alpha=0.6, label=L"$\bar{X}_n$",
         linewidth=3, legendfont=font(12))
    plot!(1:n, zeros(n), color=:black,
          linewidth=1, linestyle=:dash, label="", grid=false)
end

plot_draws()
plot_means()
