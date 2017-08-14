#=

@authors : Spencer Lyon <spencer.lyon@nyu.edu>, John Stachurski

=#

using QuantEcon
using Distributions

"""
The Lucas asset pricing model --- parameters and grid data
"""
struct LucasTree{TF<:AbstractFloat}
    gamma::TF       # coefficient of risk aversion
    beta::TF        # Discount factor in (0, 1)
    alpha::TF       # Correlation coefficient in the shock process
    sigma::TF       # Volatility of shock process
    phi::Distribution # Distribution for shock process
    grid::Vector{TF}      # Grid of points on which to evaluate prices.
    shocks::Vector{TF}    # Draws of the shock
    h::Vector{TF}         # The h function represented as a vector
end

"""
Constructor for the Lucas asset pricing model
"""
function LucasTree(;gamma::AbstractFloat=2.0,
                beta::AbstractFloat=0.95,
                alpha::AbstractFloat=0.9,
                sigma::AbstractFloat=0.1,
                grid_size::Integer=100)

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
function lucas_operator(lt::LucasTree, f::Vector)

    # == unpack names == #
    grid, alpha, beta, h = lt.grid, lt.alpha, lt.beta, lt.h
    z = lt.shocks

    Af = LinInterp(grid, f)

    Tf = [h[i] + beta *mean(Af.(grid[i]^alpha.*z)) for i in 1:length(grid)]
    return Tf
end


"""
Compute the equilibrium price function associated with Lucas tree `lt`
"""
function compute_lt_price(lt::LucasTree, max_iter::Integer=500)

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
