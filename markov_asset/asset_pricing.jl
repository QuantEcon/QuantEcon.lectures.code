#=

@authors: Spencer Lyon, Tom Sargent, John Stachurski

=#

using QuantEcon

# A default Markov chain for the state process
ρ = 0.9
σ = 0.02
n = 25
default_mc = tauchen(n, ρ, σ)

mutable struct AssetPriceModel{TF<:AbstractFloat, TI<:Integer}
    β :: TF            # Discount factor
    γ :: TF            # Coefficient of risk aversion
    mc :: MarkovChain  # State process
    n :: TI            # Number of states
    g :: Function      # Function mapping states into growth rates
end

function AssetPriceModel(;β::AbstractFloat=0.96, γ::AbstractFloat=2.0,
                          mc::MarkovChain=default_mc, g::Function=exp)
    n = size(mc.p)[1]
    return AssetPriceModel(β, γ, mc, n, g)
end


"""
Stability test for a given matrix Q.
"""
function test_stability(ap::AssetPriceModel, Q::Matrix)
    sr = maximum(abs, eigvals(Q))
    if sr >= 1 / ap.β
        msg = "Spectral radius condition failed with radius = $sr"
        throw(ArgumentError(msg))
    end
end


"""
Computes the price-dividend ratio of the Lucas tree.

"""
function tree_price(ap::AssetPriceModel)
    # == Simplify names, set up matrices  == #
    β, γ, P, y = ap.β, ap.γ, ap.mc.p, ap.mc.state_values
    y = reshape(y, 1, ap.n)
    J = P .* ap.g.(y).^(1 - γ)

    # == Make sure that a unique solution exists == #
    test_stability(ap, J)

    # == Compute v == #
    I = eye(ap.n)
    Ones = ones(ap.n)
    v = (I - β * J) \ (β * J * Ones)

    return v
end


"""
Computes price of a consol bond with payoff ζ

"""
function consol_price(ap::AssetPriceModel, ζ::AbstractFloat)
    # == Simplify names, set up matrices  == #
    β, γ, P, y = ap.β, ap.γ, ap.mc.p, ap.mc.state_values
    y = reshape(y, 1, ap.n)
    M = P .* ap.g.(y).^(-γ)

    # == Make sure that a unique solution exists == #
    test_stability(ap, M)

    # == Compute price == #
    I = eye(ap.n)
    Ones = ones(ap.n)
    p = (I - β * M) \ ( β * ζ * M * Ones)

    return p
end


"""
Computes price of a perpetual call option on a consol bond.

"""
function call_option(ap::AssetPriceModel, ζ::AbstractFloat, p_s::AbstractFloat, ϵ=1e-7)

    # == Simplify names, set up matrices  == #
    β, γ, P, y = ap.β, ap.γ, ap.mc.p, ap.mc.state_values
    y = reshape(y, 1, ap.n)
    M = P .* ap.g.(y).^(-γ)

    # == Make sure that a unique console price exists == #
    test_stability(ap, M)

    # == Compute option price == #
    p = consol_price(ap, ζ)
    w = zeros(ap.n, 1)
    error = ϵ + 1
    while (error > ϵ)
        # == Maximize across columns == #
        w_new = max.(β * M * w, p - p_s)
        # == Find maximal difference of each component and update == #
        error = maximum(abs, w - w_new)
        w = w_new
    end

    return w
end
