
#=

Filename: asset_pricing.jl

@authors: Spencer Lyon, Tom Sargent, John Stachurski

Computes asset prices with a Lucas style discount factor when the endowment
obeys geometric growth driven by a finite state Markov chain.  That is,

.. math::
    d_{t+1} = g(X_{t+1}) d_t

where

    * :math:`\{X_t\}` is a finite Markov chain with transition matrix P.

    * :math:`g` is a given positive-valued function

References
----------

    http://quant-econ.net/py/markov_asset.html

=#

using QuantEcon

# A default Markov chain for the state process
rho = 0.9
sigma = 0.02
n = 25
default_mc = tauchen(n, rho, sigma)

mutable struct AssetPriceModel{TF<:AbstractFloat, TI<:Integer}
    beta :: TF    # Discount factor
    gamma :: TF   # Coefficient of risk aversion
    mc :: MarkovChain  # State process
    n :: TI           # Number of states
    g :: Function      # Function mapping states into growth rates
end

function AssetPriceModel(;beta::AbstractFloat=0.96, gamma::AbstractFloat=2.0,
                          mc::MarkovChain=default_mc, g::Function=exp)
    n = size(mc.p)[1]
    return AssetPriceModel(beta, gamma, mc, n, g)
end


"""
Stability test for a given matrix Q.
"""
function test_stability(ap::AssetPriceModel, Q::Matrix)
    sr = maximum(abs, eigvals(Q))
    if sr >= 1 / ap.beta
        msg = "Spectral radius condition failed with radius = $sr"
        throw(ArgumentError(msg))
    end
end


"""
Computes the price-dividend ratio of the Lucas tree.

"""
function tree_price(ap::AssetPriceModel)
    # == Simplify names, set up matrices  == #
    beta, gamma, P, y = ap.beta, ap.gamma, ap.mc.p, ap.mc.state_values
    y = reshape(y, 1, ap.n)
    J = P .* ap.g.(y).^(1 - gamma)

    # == Make sure that a unique solution exists == #
    test_stability(ap, J)

    # == Compute v == #
    I = eye(ap.n)
    Ones = ones(ap.n)
    v = (I - beta * J) \ (beta * J * Ones)

    return v
end


"""
Computes price of a consol bond with payoff zeta

"""
function consol_price(ap::AssetPriceModel, zeta::AbstractFloat)
    # == Simplify names, set up matrices  == #
    beta, gamma, P, y = ap.beta, ap.gamma, ap.mc.p, ap.mc.state_values
    y = reshape(y, 1, ap.n)
    M = P .* ap.g.(y).^(- gamma)

    # == Make sure that a unique solution exists == #
    test_stability(ap, M)

    # == Compute price == #
    I = eye(ap.n)
    Ones = ones(ap.n)
    p = (I - beta * M) \ ( beta * zeta * M * Ones)

    return p
end


"""
Computes price of a perpetual call option on a consol bond.

"""
function call_option(ap::AssetPriceModel, zeta::AbstractFloat, p_s::AbstractFloat, epsilon=1e-7)

    # == Simplify names, set up matrices  == #
    beta, gamma, P, y = ap.beta, ap.gamma, ap.mc.p, ap.mc.state_values
    y = reshape(y, 1, ap.n)
    M = P .* ap.g.(y).^(- gamma)

    # == Make sure that a unique console price exists == #
    test_stability(ap, M)

    # == Compute option price == #
    p = consol_price(ap, zeta)
    w = zeros(ap.n, 1)
    error = epsilon + 1
    while (error > epsilon)
        # == Maximize across columns == #
        w_new = max.(beta * M * w, p - p_s)
        # == Find maximal difference of each component and update == #
        error = maximum(abs, w-w_new)
        w = w_new
    end

    return w
end
