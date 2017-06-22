#=
Solving the optimal growth problem via Coleman policy function iteration. 
Author: Shunsuke Hori

=#

using QuantEcon

"""
g: input policy function
grid: grid points
beta: discount factor
u_prime: derivative of utility function
f: production function
f_prime: derivative of production function
shocks::shock draws, used for Monte Carlo integration to compute expectation
Kg: output value is stored
"""
function coleman_operator!(g::Vector{Float64},
                           grid::Vector{Float64},
                           beta::Float64,
                           u_prime::Function,
                           f::Function,
                           f_prime::Function,
                           shocks::Vector{Float64},
                           Kg::Vector{Float64}=similar(g))
    
    # This function requires the container of the output value as argument Kg

    # Construct linear interpolation object #
    g_func=LinInterp(grid, g)    

    # solve for updated consumption value #
    for (i,y) in enumerate(grid)
        function h(c)
            vals = u_prime.(g_func.(f(y - c)*shocks)).*f_prime(y - c).*shocks
            return u_prime(c) - beta * mean(vals)
        end
        Kg[i] = brent(h, 1e-10, y-1e-10)
    end
    return Kg
end

# The following function does NOT require the container of the output value as argument
function coleman_operator(g::Vector{Float64},
                          grid::Vector{Float64},
                          beta::Float64,
                          u_prime::Function,
                          f::Function,
                          f_prime::Function,
                          shocks::Vector{Float64})

    return coleman_operator!(g, grid, beta, u_prime, 
                             f, f_prime, shocks, similar(g))
end
