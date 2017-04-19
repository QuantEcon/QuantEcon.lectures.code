#=
Filename: aiyagari_compute_equilibrium.jl
Author: Victoria Gregory
Date: 8/30/2016

Draws the aggregate supply and demand curves for
the Aiyagari model.
=#


# Firms' parameters
A = 1
N = 1
alpha = 0.33
beta = 0.96
delta = 0.05

"""
Compute wage rate given an interest rate, r
"""
function r_to_w(r::Float64)
    return A * (1 - alpha) * (A * alpha / (r + delta)) ^ (alpha / (1 - alpha))
end

"""
Inverse demand curve for capital. The interest rate 
associated with a given demand for capital K.
"""
function rd(K::Float64)
    return A * alpha * (N / K) ^ (1 - alpha) - delta
end

"""
Map prices to the induced level of capital stock.

##### Arguments
- `am::Household` : Household instance for problem we want to solve
- `r::Float64` : interest rate

##### Returns
- The implied level of aggregate capital
"""
function prices_to_capital_stock(am::Household, r::Float64)
    
    # Set up problem
    w = r_to_w(r)
    setup_R!(am, r, w)
    aiyagari_ddp = DiscreteDP(am.R, am.Q, am.beta)

    # Compute the optimal policy
    results = solve(aiyagari_ddp, PFI)

    # Compute the stationary distribution
    stationary_probs = stationary_distributions(results.mc)[:, 1][1]
    
    # Return K
    return sum(am.s_vals[:, 1] .* stationary_probs)
end

# Create an instance of Household
z_chain = MarkovChain([0.67 0.33; 0.33 0.67], [0.5, 1.5])
am = Household(z_chain=z_chain, beta=beta, a_max=20.0)

# Create a grid of r values at which to compute demand and supply of capital
num_points = 20
r_vals = linspace(0.02, 1/beta - 1, num_points)

# Compute supply of capital
k_vals = Array{Float64}(num_points, 1)
for i in 1:num_points
    k_vals[i] = prices_to_capital_stock(am, r_vals[i])
end

# Plot against demand for capital by firms
demand = [rd(k) for k in k_vals]
labels =  ["demand for capital"; "supply of capital"]
plot(k_vals, [demand r_vals], label=labels', lw=2, alpha=0.6)
plot!(xlabel="capital", ylabel="interest rate")
