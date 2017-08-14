#=

Author: Shunsuke Hori

=#

struct Model{TF <: AbstractFloat, TR <: Real, TI <: Integer}
    alpha::TR    # Productivity parameter
    beta::TF     # Discount factor
    gamma::TR    # risk aversion
    mu::TR       # First parameter in lognorm(mu, sigma)
    s::TR        # Second parameter in lognorm(mu, sigma)
    grid_min::TR # Smallest grid point
    grid_max::TR # Largest grid point
    grid_size::TI # Number of grid points
    u::Function       # utility function
    u_prime::Function # derivative of utility function
    f::Function       # production function
    f_prime::Function # derivative of production function
    grid::Vector{TR}   # grid
end

"""
construct Model instance using the information of parameters and functional form

arguments: see above

return: Model type instance
"""
function Model(; alpha::Real=0.65,   # Productivity parameter
                 beta::AbstractFloat=0.95,    # Discount factor
                 gamma::Real=1.0,    # risk aversion
                 mu::Real=0.0,       # First parameter in lognorm(mu, sigma)
                 s::Real=0.1,        # Second parameter in lognorm(mu, sigma)
                 grid_min::Real=1e-6,# Smallest grid point
                 grid_max::Real=4.0, # Largest grid point
                 grid_size::Integer=200, # Number of grid points
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
