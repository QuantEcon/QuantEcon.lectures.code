#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>

=#

using QuantEcon
using Interpolations
using Distributions

"""
Unemployment/search problem where offer distribution is unknown

##### Fields

- `bet::Real` : Discount factor on (0, 1)
- `c::Real` : Unemployment compensation
- `F::Distribution` : Offer distribution `F`
- `G::Distribution` : Offer distribution `G`
- `f::Function` : The pdf of `F`
- `g::Function` : The pdf of `G`
- `n_w::Int` : Number of points on the grid for w
- `w_max::Real` : Maximum wage offer
- `w_grid::AbstractVector` : Grid of wage offers w
- `n_pi::Int` : Number of points on grid for pi
- `pi_min::Real` : Minimum of pi grid
- `pi_max::Real` : Maximum of pi grid
- `pi_grid::AbstractVector` : Grid of probabilities pi
- `quad_nodes::Vector` : Notes for quadrature ofer offers
- `quad_weights::Vector` : Weights for quadrature ofer offers

"""
struct SearchProblem{TR<:Real, TI<:Integer, TF<:AbstractFloat,
                    TAVw<:AbstractVector{TF}, TAVpi<:AbstractVector{TF}}
    bet::TR
    c::TR
    F::Distribution
    G::Distribution
    f::Function
    g::Function
    n_w::TI
    w_max::TR
    w_grid::TAVw
    n_pi::TI
    pi_min::TR
    pi_max::TR
    pi_grid::TAVpi
    quad_nodes::Vector{TF}
    quad_weights::Vector{TF}
end

"""
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Integer(40)` : Number of points in wage offer grid
- `pi_grid_size::Integer(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter

"""

# use key word argment
function SearchProblem(;bet::Real=0.95, c::Real=0.6, F_a::Real=1, F_b::Real=1,
                       G_a::Real=3, G_b::Real=1.2, w_max::Real=2.0,
                       w_grid_size::Integer=40, pi_grid_size::Integer=40)

    F = Beta(F_a, F_b)
    G = Beta(G_a, G_b)

    # NOTE: the x./w_max)./w_max in these functions makes our dist match
    #       the scipy one with scale=w_max given
    f(x) = pdf(F, x/w_max)/w_max
    g(x) = pdf(G, x/w_max)/w_max

    pi_min = 1e-3  # avoids instability
    pi_max = 1 - pi_min

    w_grid = linspace(0, w_max, w_grid_size)
    pi_grid = linspace(pi_min, pi_max, pi_grid_size)

    nodes, weights = qnwlege(21, 0.0, w_max)

    SearchProblem(bet, c, F, G, f, g,
                  w_grid_size, w_max, w_grid,
                  pi_grid_size, pi_min, pi_max, pi_grid, nodes, weights)
end

function q(sp::SearchProblem, w, pi_val)
    new_pi = 1.0 / (1 + ((1 - pi_val) * sp.g(w)) / (pi_val * sp.f(w)))

    # Return new_pi when in [pi_min, pi_max] and else end points
    return clamp(new_pi, sp.pi_min, sp.pi_max)
end

"""
Apply the Bellman operator for a given model and initial value.

##### Arguments

- `sp::SearchProblem` : Instance of `SearchProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output.
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.

"""
function bellman_operator!(sp::SearchProblem, v::Matrix, out::Matrix;
                          ret_policy::Bool=false)
    # Simplify names
    f, g, bet, c = sp.f, sp.g, sp.bet, sp.c
    nodes, weights = sp.quad_nodes, sp.quad_weights

    vf = extrapolate(interpolate((sp.w_grid, sp.pi_grid), v,
                     Gridded(Linear())), Flat())

    # set up quadrature nodes/weights
    # q_nodes, q_weights = qnwlege(21, 0.0, sp.w_max)

    for (w_i, w) in enumerate(sp.w_grid)
        # calculate v1
        v1 = w / (1 - bet)

        for (pi_j, _pi) in enumerate(sp.pi_grid)
            # calculate v2
            integrand(m) = [vf[m[i], q.(sp, m[i], _pi)]*
                          (_pi*f(m[i])+(1-_pi)*g(m[i])) for i in 1:length(m)]
            integral = do_quad(integrand, nodes, weights)
            # integral = do_quad(integrand, q_nodes, q_weights)
            v2 = c + bet * integral

            # return policy if asked for, otherwise return max of values
            out[w_i, pi_j] = ret_policy ? v1 > v2 : max(v1, v2)
        end
    end
    return out
end

function bellman_operator(sp::SearchProblem, v::Matrix;
                          ret_policy::Bool=false)
    out_type = ret_policy ? Bool : Float64
    out = Array{out_type}(sp.n_w, sp.n_pi)
    bellman_operator!(sp, v, out, ret_policy=ret_policy)
end


"""
Extract the greedy policy (policy function) of the model.

##### Arguments

- `sp::SearchProblem` : Instance of `SearchProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function

"""
get_greedy!(sp::SearchProblem, v::Matrix, out::Matrix) =
    bellman_operator!(sp, v, out, ret_policy=true)

get_greedy(sp::SearchProblem, v::Matrix) =
    bellman_operator(sp, v, ret_policy=true)

"""
Updates the reservation wage function guess phi via the operator Q.

##### Arguments

- `sp::SearchProblem` : Instance of `SearchProblem`
- `phi::Vector`: Current guess for phi
- `out::Vector` : Storage for output

##### Returns

None, `out` is updated in place to hold the updated levels of phi
"""
function res_wage_operator!(sp::SearchProblem, phi::Vector, out::Vector)
    # Simplify name
    f, g, bet, c = sp.f, sp.g, sp.bet, sp.c

    # Construct interpolator over pi_grid, given phi
    phi_f = LinInterp(sp.pi_grid, phi)

    # set up quadrature nodes/weights
    q_nodes, q_weights = qnwlege(7, 0.0, sp.w_max)

    for (i, _pi) in enumerate(sp.pi_grid)
        integrand(x) = max.(x, phi_f.(q.(sp, x, _pi))).*(_pi*f(x) + (1-_pi)*g(x))
        integral = do_quad(integrand, q_nodes, q_weights)
        out[i] = (1 - bet)*c + bet*integral
    end
end

"""
Updates the reservation wage function guess phi via the operator Q.

See the documentation for the mutating method of this function for more details
on arguments
"""
function res_wage_operator(sp::SearchProblem, phi::Vector)
    out = similar(phi)
    res_wage_operator!(sp, phi, out)
    return out
end
