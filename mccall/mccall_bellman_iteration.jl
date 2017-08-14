using Distributions

# A default utility function

function u(c::Real, sigma::Real)
    if c > 0
        return (c^(1 - sigma) - 1) / (1 - sigma)
    else
        return -10e6
    end
end

# default wage vector with probabilities

const n = 60                                   # n possible outcomes for wage
const default_w_vec = linspace(10, 20, n)   # wages between 10 and 20
const a, b = 600, 400                          # shape parameters
const dist = BetaBinomial(n-1, a, b)
const default_p_vec = pdf(dist)

mutable struct McCallModel{TF <: AbstractFloat,
                           TAV <: AbstractVector{TF},
                           TAV2 <: AbstractVector{TF}}
    alpha::TF        # Job separation rate
    beta::TF         # Discount rate
    gamma::TF        # Job offer rate
    c::TF            # Unemployment compensation
    sigma::TF        # Utility parameter
    w_vec::TAV # Possible wage values
    p_vec::TAV2 # Probabilities over w_vec

    McCallModel(alpha::TF=0.2,
                beta::TF=0.98,
                gamma::TF=0.7,
                c::TF=6.0,
                sigma::TF=2.0,
                w_vec::TAV=default_w_vec,
                p_vec::TAV2=default_p_vec) where {TF, TAV, TAV2} =
        new{TF, TAV, TAV2}(alpha, beta, gamma, c, sigma, w_vec, p_vec)
end

"""
A function to update the Bellman equations.  Note that V_new is modified in
place (i.e, modified by this function).  The new value of U is returned.

"""
function update_bellman!(mcm::McCallModel, V::AbstractVector,
                         V_new::AbstractVector, U::Real)
    # Simplify notation
    alpha, beta, sigma, c, gamma = mcm.alpha, mcm.beta, mcm.sigma, mcm.c, mcm.gamma

    for (w_idx, w) in enumerate(mcm.w_vec)
        # w_idx indexes the vector of possible wages
        V_new[w_idx] = u(w, sigma) + beta * ((1 - alpha) * V[w_idx] + alpha * U)
    end

    U_new = u(c, sigma) + beta * (1 - gamma) * U +
                    beta * gamma * dot(max.(U, V), mcm.p_vec)

    return U_new
end


function solve_mccall_model(mcm::McCallModel;
                            tol::AbstractFloat=1e-5, max_iter::Integer=2000)

    V = ones(length(mcm.w_vec))  # Initial guess of V
    V_new = similar(V)           # To store updates to V
    U = 1.0                        # Initial guess of U
    i = 0
    error = tol + 1

    while error > tol && i < max_iter
        U_new = update_bellman!(mcm, V, V_new, U)
        error_1 = maximum(abs, V_new - V)
        error_2 = abs(U_new - U)
        error = max(error_1, error_2)
        V[:] = V_new
        U = U_new
        i += 1
    end

    return V, U
end
