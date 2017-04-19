#=
Filename: aiyagari_household.jl
Author: Victoria Gregory
Date: 8/29/2016

This file defines the Household type (and its constructor)
for setting up an Aiyagari household problem.
=#

using QuantEcon

"""
Stores all the parameters that define the household's
problem.

##### Fields

- `r::Float64` : interest rate
- `w::Float64` : wage
- `beta::Float64` : discount factor
- `z_chain::MarkovChain` : MarkovChain for income
- `a_min::Float64` : minimum on asset grid
- `a_max::Float64` : maximum on asset grid
- `a_size::Int64` : number of points on asset grid
- `z_size::Int64` : number of points on income grid
- `n::Int64` : number of points in state space: (a, z)
- `s_vals::Array{Float64}` : stores all the possible (a, z) combinations
- `s_i_vals::Array{Int64}` : stores indices of all the possible (a, z) combinations
- `R::Array{Float64}` : reward array
- `Q::Array{Float64}` : transition probability array
"""
type Household
    r::Float64
    w::Float64
    beta::Float64
    z_chain::MarkovChain{Float64,Array{Float64,2},Array{Float64,1}}
    a_min::Float64
    a_max::Float64
    a_size::Int64
    a_vals::Vector{Float64}
    z_size::Int64
    n::Int64
    s_vals::Array{Float64}
    s_i_vals::Array{Int64}
    R::Array{Float64}
    Q::Array{Float64}
end

"""
Constructor for `Household`

##### Arguments
- `r::Float64(0.01)` : interest rate
- `w::Float64(1.0)` : wage
- `beta::Float64(0.96)` : discount factor
- `z_chain::MarkovChain` : MarkovChain for income
- `a_min::Float64(1e-10)` : minimum on asset grid
- `a_max::Float64(18.0)` : maximum on asset grid
- `a_size::Int64(200)` : number of points on asset grid

"""
function Household(;r::Float64=0.01, w::Float64=1.0, beta::Float64=0.96, 
                   z_chain::MarkovChain{Float64,Array{Float64,2},Array{Float64,1}}
                   =MarkovChain([0.9 0.1; 0.1 0.9], [0.1; 1.0]), a_min::Float64=1e-10, 
                   a_max::Float64=18.0, a_size::Int64=200)
    
    # set up grids
    a_vals = linspace(a_min, a_max, a_size)
    z_size = length(z_chain.state_values)
    n = a_size*z_size
    s_vals = gridmake(a_vals, z_chain.state_values)
    s_i_vals = gridmake(1:a_size, 1:z_size)

    # set up Q
    Q = zeros(Float64, n, a_size, n)
    for next_s_i in 1:n
        for a_i in 1:a_size
            for s_i in 1:n
                z_i = s_i_vals[s_i, 2]
                next_z_i = s_i_vals[next_s_i, 2]
                next_a_i = s_i_vals[next_s_i, 1]
                if next_a_i == a_i
                    Q[s_i, a_i, next_s_i] = z_chain.p[z_i, next_z_i]
                end
            end
        end
    end

    # placeholder for R
    R = fill(-Inf, n, a_size)
    h = Household(r, w, beta, z_chain, a_min, a_max, a_size, 
                  a_vals, z_size, n, s_vals, s_i_vals, R, Q)

    setup_R!(h, r, w)

    return h

end

"""
Update the reward array of a Household object, given
a new interest rate and wage.

##### Arguments
- `h::Household` : instance of Household type
- `r::Float64(0.01)` : interest rate
- `w::Float64(1.0)` : wage
"""
function setup_R!(h::Household, r::Float64, w::Float64)

    # set up R
    R = h.R
    for new_a_i in 1:h.a_size
        a_new = h.a_vals[new_a_i]
        for s_i in 1:h.n
            a = h.s_vals[s_i, 1]
            z = h.s_vals[s_i, 2]
            c = w * z + (1 + r) * a - a_new
            if c > 0
                R[s_i, new_a_i] = log(c)
            end
        end
    end

    h.r = r
    h.w = w
    h.R = R
    h

end