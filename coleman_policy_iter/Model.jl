#=

Filename: Model.jl.
Author: Shunsuke Hori

Defining an instance "Model" storing parameters, functional forms, and grid.

=#

type Model
    alpha::Float64    # Productivity parameter
    beta::Float64     # Discount factor
    gamma::Float64    # risk aversion
    mu::Float64       # First parameter in lognorm(mu, sigma)
    s::Float64        # Second parameter in lognorm(mu, sigma)
    grid_min::Float64 # Smallest grid point
    grid_max::Float64 # Largest grid point
    grid_size::Signed # Number of grid points
    u::Function       # utility function
    u_prime::Function # derivative of utility function
    f::Function       # production function
    f_prime::Function # derivative of production function
    grid::Vector{Float64}   # grid 
end

"""
construct Model instance using the information of parameters and functional form

arguments: see above

return: Model type instance
"""
function Model(; alpha::Float64=0.65,   # Productivity parameter
                 beta::Float64=0.95,    # Discount factor
                 gamma::Float64=1.0,    # risk aversion
                 mu::Float64=0.0,       # First parameter in lognorm(mu, sigma)
                 s::Float64=0.1,        # Second parameter in lognorm(mu, sigma)
                 grid_min::Float64=1e-6,# Smallest grid point
                 grid_max::Float64=4.0, # Largest grid point
                 grid_size::Signed=200, # Number of grid points
                 u::Function= c->(c^(1-gamma)-1)/(1-gamma), # utility function
                 u_prime::Function = c-> c^(-gamma),        # u'
                 f::Function = k-> k^alpha,                 # production function
                 f_prime::Function = k -> alpha*k^(alpha-1) # f'
                 )

    grid=collect(linspace(grid_min, grid_max, grid_size))

    if gamma == 1   # when gamma==1, log utility is assigned
        u_log(c) = log(c)
        m = Model(alpha, beta, gamma, mu, s, grid_min, grid_max, 
                grid_size, u_log, u_prime, f, f_prime, grid)
    else
        m = Model(alpha, beta, gamma, mu, s, grid_min, grid_max, 
                  grid_size, u, u_prime, f, f_prime, grid)
    end
    return m
end

