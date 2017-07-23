
#=
Visual illustration of the central limit theorem in 3d

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

References
----------
Based off the original python  file clt3d.py
=#
using Plots
pyplot()
using Distributions
using KernelDensity

beta_dist = Beta(2.0, 2.0)


function gen_x_draws(k)
    bdraws = rand(beta_dist, 3, k)

    # == Transform rows, so each represents a different distribution == #
    bdraws[1, :] -= 0.5
    bdraws[2, :] += 0.6
    bdraws[3, :] -= 1.1

    # == Set X[i] = bdraws[j, i], where j is a random draw from {1, 2, 3} == #
    js = rand(1:3, k)
    X = Array{Float64}(k)
    for i=1:k
        X[i]=  bdraws[js[i], i]
    end

    # == Rescale, so that the random variable is zero mean == #
    m, sigma = mean(X), std(X)
    return (X .- m) ./ sigma
end

nmax = 5
reps = 100000
ns = 1:nmax

# == Form a matrix Z such that each column is reps independent draws of X == #
Z = Array{Float64}(reps, nmax)
for i=ns
    Z[:, i] = gen_x_draws(reps)
end

# == Take cumulative sum across columns
S = cumsum(Z, 2)

# == Multiply j-th column by sqrt j == #
Y = S .* (1. ./ sqrt.(ns))'

# == Plot == #
a, b = -3, 3
gs = 100
xs = linspace(a, b, gs)

x_vec = []
y_vec = []
z_vec = []
colors = []
for n=ns
    kde_est = kde(Y[:, n])
    _xs, ys = kde_est.x, kde_est.density
    push!(x_vec, collect(_xs))
    push!(y_vec, ys)
    push!(z_vec, collect(n*ones( length(_xs))))
    push!(colors, RGBA(0, 0, 0, 1-(n-1)/nmax))
end

plot(x_vec, z_vec, y_vec, color=reshape(colors,1,length(colors)), legend=:none)
plot!(xlims=(a,b), xticks=[-3; 0; 3])
plot!(ylims=(1, nmax), yticks=ns, ylabel="n")
plot!(zlims=(0, 0.4), zticks=[0.2; 0.4])
