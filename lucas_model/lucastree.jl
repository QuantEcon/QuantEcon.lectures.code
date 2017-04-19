
#=

Solves the price function for the Lucas tree in a continuous state
setting, using piecewise linear approximation for the sequence of
candidate price functions.  The consumption endownment follows the
log linear AR(1) process

    log y' = alpha log y + sigma epsilon

where y' is a next period y and epsilon is an iid standard normal
shock. Hence

    y' = y^alpha * xi   where xi = e^(sigma * epsilon)

The distribution phi of xi is

    phi = LN(0, sigma^2) where LN means lognormal

@authors : Spencer Lyon <spencer.lyon@nyu.edu>, John Stachurski


References
----------

http://quant-econ.net/jl/markov_asset.html

=#

using QuantEcon
using Distributions


"""
A function that takes two arrays and returns a function that approximates the
data using continuous piecewise linear interpolation.

"""
function lin_interp(x_vals::Vector{Float64}, y_vals::Vector{Float64})
    # == linear interpolation inside grid, constant values outside grid == #
    w = LinInterp(x_vals, y_vals)
    return w
end



"""
The Lucas asset pricing model --- parameters and grid data
"""
type LucasTree
    gamma::Real       # coefficient of risk aversion 
    beta::Real        # Discount factor in (0, 1)
    alpha::Real       # Correlation coefficient in the shock process
    sigma::Real       # Volatility of shock process
    phi::Distribution # Distribution for shock process
    grid::Vector      # Grid of points on which to evaluate prices.
    shocks::Vector    # Draws of the shock
    h::Vector         # The h function represented as a vector
end



"""
Constructor for the Lucas asset pricing model 
"""
function LucasTree(;gamma=2.0, 
                beta=0.95, 
                alpha=0.9, 
                sigma=0.1,
                grid_size=100)

    phi = LogNormal(0.0, sigma)
    shocks = rand(phi, 500)

    # == build a grid with mass around stationary distribution == #
    ssd = sigma / sqrt(1 - alpha^2)
    grid_min, grid_max = exp(-4 * ssd), exp(4 * ssd)
    grid = collect(linspace(grid_min, grid_max, grid_size))

    # == set h(y) = beta * int u'(G(y,z)) G(y,z) phi(dz) == #
    h = similar(grid)
    for (i, y) in enumerate(grid)
        h[i] = beta * mean((y^alpha .* shocks).^(1 - gamma))
    end

    return LucasTree(gamma,
                    beta,
                    alpha,
                    sigma,
                    phi,
                    grid,
                    shocks,
                    h)
end


"""
The approximate Lucas operator, which computes and returns updated function
Tf on the grid points.
"""
function lucas_operator(lt::LucasTree, f::Vector{Float64})

    # == unpack names == #
    grid, alpha, beta, h = lt.grid, lt.alpha, lt.beta, lt.h
    z = lt.shocks

    Tf = similar(f)
    Af = lin_interp(grid, f)

    for (i, y) in enumerate(grid)
        Tf[i] = h[i] + beta * mean(Af.(y^alpha .* z))
    end
    return Tf
end


"""
Compute the equilibrium price function associated with Lucas tree `lt`
"""
function compute_lt_price(lt::LucasTree, max_iter=500)

    # == Simplify names == #
    grid = lt.grid
    alpha, beta, gamma = lt.alpha, lt.beta, lt.gamma

    # == Create suitable initial vector to iterate from == # 
    f_init = zeros(grid)  

    func(f_vec) = lucas_operator(lt, f_vec)
    f = compute_fixed_point(func, f_init; 
                                    max_iter=max_iter, 
                                    err_tol=1e-4, 
                                    verbose=false)

    # p(y) = f(y) * y^gamma
    price = f .* grid.^(gamma)

    return price
end

