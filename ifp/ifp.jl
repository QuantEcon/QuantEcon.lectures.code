
#=
Tools for solving the standard optimal savings / income fluctuation
problem for an infinitely lived consumer facing an exogenous income
process that evolves according to a Markov chain.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-18

References
----------

http://quant-econ.net/jl/ifp.html

=#
using Interpolations
using Optim

# utility and marginal utility functions
u(x) = log(x)
du(x) = 1 ./ x

"""
Income fluctuation problem

##### Fields

- `r::Float64` : Strictly positive interest rate
- `R::Float64` : The interest rate plus 1 (strictly greater than 1)
- `bet::Float64` : Discount rate in (0, 1)
- `b::Float64` :  The borrowing constraint
- `Pi::Matrix{Floa64}` : Transition matrix for `z`
- `z_vals::Vector{Float64}` : Levels of productivity
- `asset_grid::LinSpace{Float64}` : Grid of asset values

"""
type ConsumerProblem
    r::Float64
    R::Float64
    bet::Float64
    b::Float64
    Pi::Matrix{Float64}
    z_vals::Vector{Float64}
    asset_grid::LinSpace{Float64}
end

function ConsumerProblem(;r=0.01, bet=0.96, Pi=[0.6 0.4; 0.05 0.95],
                         z_vals=[0.5, 1.0], b=0.0, grid_max=16, grid_size=50)
    R = 1 + r
    asset_grid = linspace(-b, grid_max, grid_size)

    ConsumerProblem(r, R, bet, b, Pi, z_vals, asset_grid)
end

"""
Given a matrix of size `(length(cp.asset_grid), length(cp.z_vals))`, construct
an interpolation object that does linear interpolation in the asset dimension
and has a lookup table in the z dimension
"""
function Interpolations.interpolate(cp::ConsumerProblem, x::AbstractMatrix)
    sz = (length(cp.asset_grid), length(cp.z_vals))
    if size(x) != sz
        msg = "x must have dimensions $(sz)"
        throw(DimensionMismatch(msg))
    end

    itp = interpolate(x, (BSpline(Linear()), NoInterp()), OnGrid())
    scale(itp, cp.asset_grid, 1:sz[2])
end

"""
Apply the Bellman operator for a given model and initial value.

##### Arguments

- `cp::ConsumerProblem` : Instance of `ConsumerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.

"""
function update_bellman!(cp::ConsumerProblem, V::Matrix, out::Matrix;
                           ret_policy::Bool=false)
    # simplify names, set up arrays
    R, Pi, bet, b = cp.R, cp.Pi, cp.bet, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals

    z_idx = 1:length(z_vals)
    vf = interpolate(cp, V)

    # compute lower_bound for optimization
    opt_lb = 1e-8

    # solve for RHS of Bellman equation
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)

            function obj(c)
                y = 0.0
                for j in z_idx
                    y += vf[R*a+z-c, j] * Pi[i_z, j]
                end
                return -u(c)  - bet * y
            end
            res = optimize(obj, opt_lb, R.*a.+z.+b)
            c_star = Optim.minimizer(res)

            if ret_policy
                out[i_a, i_z] = c_star
            else
               out[i_a, i_z] = - obj(c_star)
            end

        end
    end
    out
end

update_bellman(cp::ConsumerProblem, V::Matrix; ret_policy=false) =
    update_bellman!(cp, V, similar(V); ret_policy=ret_policy)

"""
Extract the greedy policy (policy function) of the model.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function

"""
get_greedy!(cp::ConsumerProblem, V::Matrix, out::Matrix) =
    update_bellman!(cp, V, out, ret_policy=true)

get_greedy(cp::ConsumerProblem, V::Matrix) =
    update_bellman(cp, V, ret_policy=true)

"""
The approximate Coleman operator.

Iteration with this operator corresponds to policy function
iteration. Computes and returns the updated consumption policy
c.  The array c is replaced with a function cf that implements
univariate linear interpolation over the asset grid for each
possible value of z.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `c::Matrix`: Current guess for the policy function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function

"""
function update_coleman!(cp::ConsumerProblem, c::Matrix, out::Matrix)
    # simplify names, set up arrays
    R, Pi, bet, b = cp.R, cp.Pi, cp.bet, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    z_size = length(z_vals)
    gam = R * bet
    vals = Array{Float64}(z_size)

    cf = interpolate(cp, c)

    # linear interpolation to get consumption function. Updates vals inplace
    cf!(a, vals) = map!(i->cf[a, i], vals, 1:z_size)

    # compute lower_bound for optimization
    opt_lb = 1e-8

    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)
            function h(t)
                cf!(R*a+z-t, vals)  # update vals
                expectation = dot(du(vals), vec(Pi[i_z, :]))
                return abs(du(t) - max(gam * expectation, du(R*a+z+b)))
            end
            opt_ub = R*a + z + b  # addresses issue #8 on github
            res = optimize(h, min(opt_lb, opt_ub - 1e-2), opt_ub,
                           method=Optim.Brent())
            out[i_a, i_z] = Optim.minimizer(res)
        end
    end
    out
end


"""
Apply the Coleman operator for a given model and initial value

See the specific methods of the mutating version of this function for more
details on arguments
"""
update_coleman(cp::ConsumerProblem, c::Matrix) =
    update_coleman!(cp, c, similar(c))

function init_values(cp::ConsumerProblem)
    # simplify names, set up arrays
    R, bet, b = cp.R, cp.bet, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    shape = length(asset_grid), length(z_vals)
    V, c = Array{Float64}(shape...), Array{Float64}(shape...)

    # Populate V and c
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)
            c_max = R*a + z + b
            c[i_a, i_z] = c_max
            V[i_a, i_z] = u(c_max) ./ (1 - bet)
        end
    end

    return V, c
end


