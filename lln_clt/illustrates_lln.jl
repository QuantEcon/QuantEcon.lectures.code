#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#
using Plots
pyplot()
using Distributions
using LaTeXStrings

n = 100
srand(42)  # reproducible results

# == Arbitrary collection of distributions == #
distributions = Dict("student's t with 10 degrees of freedom" => TDist(10),
                 "beta(2, 2)" => Beta(2.0, 2.0),
                 "lognormal LN(0, 1/2)" => LogNormal(0.5),
                 "gamma(5, 1/2)" => Gamma(5.0, 2.0),
                 "poisson(4)" => Poisson(4),
                 "exponential with lambda = 1" => Exponential(1))

num_plots = 3
dist_data = zeros(num_plots, n)
sample_means = []
dist_means = []
titles = []
for i = 1:num_plots
    dist_names = collect(keys(distributions))
    # == Choose a randomly selected distribution == #
    name = dist_names[rand(1:length(dist_names))]
    dist = pop!(distributions, name)

    # == Generate n draws from the distribution == #
    data = rand(dist, n)

    # == Compute sample mean at each n == #
    sample_mean = Array{Float64}(n)
    for j=1:n
        sample_mean[j] = mean(data[1:j])
    end

    m = mean(dist)

    dist_data[i, :] = data'
    push!(sample_means, sample_mean)
    push!(dist_means, m*ones(n))
    push!(titles, name)

end

# == Plot == #
N = repmat(reshape(repmat(1:n, 1, num_plots)', 1, n*num_plots), 2, 1)
heights = [zeros(1,n*num_plots); reshape(dist_data, 1, n*num_plots)]
plot(N, heights, layout=(3, 1), label="", color=:grey, alpha=0.5)
plot!(1:n, dist_data', layout=(3, 1), color=:grey, markershape=:circle,
        alpha=0.5, label="", linewidth=0)
plot!(1:n, sample_means, linewidth=3, alpha=0.6, color=:green, legend=:topleft,
      layout=(3, 1), label=[LaTeXString("\$\\bar{X}_n\$") "" ""])
plot!(1:n, dist_means, color=:black, linewidth=1.5, layout=(3, 1),
      linestyle=:dash, grid=false, label=[LaTeXString("\$\\mu\$") "" ""])
plot!(title=reshape(titles, 1, length(titles)))
