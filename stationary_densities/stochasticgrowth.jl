#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#
using QuantEcon: LAE, lae_est
using Distributions

s = 0.2
delta = 0.1
a_sigma = 0.4  # A = exp(B) where B ~ N(0, a_sigma)
alpha = 0.4  # We set f(k) = k**alpha
psi_0 = Beta(5.0, 5.0)  # Initial distribution
phi = LogNormal(0.0, a_sigma)


function p(x, y)
    #=
    Stochastic kernel for the growth model with Cobb-Douglas production.
    Both x and y must be strictly positive.
    =#
    d = s * x.^alpha

    # scipy silently evaluates the pdf of the lognormal dist at a negative
    # value as zero. It should be undefined and Julia recognizes this.
    pdf_arg = clamp.((y .- (1-delta) .* x) ./ d, eps(), Inf)
    return pdf(phi, pdf_arg) ./ d
end


n = 10000  # Number of observations at each date t
T = 30  # Compute density of k_t at 1,...,T+1

# Generate matrix s.t. t-th column is n observations of k_t
k = Array{Float64}(n, T)
A = rand!(phi, Array{Float64}(n, T))

# Draw first column from initial distribution
k[:, 1] = rand(psi_0, n) ./ 2  # divide by 2 to match scale=0.5 in py version
for t=1:T-1
    k[:, t+1] = s*A[:, t] .* k[:, t].^alpha + (1-delta) .* k[:, t]
end

# Generate T instances of LAE using this data, one for each date t
laes = [LAE(p, k[:, t]) for t=T:-1:1]

# Plot
ygrid = linspace(0.01, 4.0, 200)
laes_plot = []
colors = []
for i = 1:T
    psi = laes[i]
    push!(laes_plot, lae_est(psi, ygrid))
    push!(colors,  RGBA(0, 0, 0, 1 - (i - 1)/T))
end
plot(ygrid, laes_plot, color=reshape(colors, 1, length(colors)), lw=2, xlabel="capital", legend=:none)
t=LaTeXString("Density of \$k_1\$ (lighter) to \$k_T\$ (darker) for \$T=$T\$")
plot!(title=t)
