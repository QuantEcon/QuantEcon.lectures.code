
#=

Implements iteration on the Bellman equations to solve the McCall growth model

=#

using Distributions

# A default utility function

function u(c, sigma)
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


type McCallModel
    alpha::Float64        # Job separation rate
    beta::Float64         # Discount rate
    gamma::Float64        # Job offer rate
    c::Float64            # Unemployment compensation
    sigma::Float64        # Utility parameter
    w_vec::Vector{Float64} # Possible wage values
    p_vec::Vector{Float64} # Probabilities over w_vec

    function McCallModel(alpha=0.2,
                         beta=0.98,
                         gamma=0.7,
                         c=6.0,
                         sigma=2.0,
                         w_vec=default_w_vec,
                         p_vec=default_p_vec)

        return new(alpha, beta, gamma, c, sigma, w_vec, p_vec)
    end
end



"""
A function to update the Bellman equations.  Note that V_new is modified in
place (i.e, modified by this function).  The new value of U is returned.

"""
function update_bellman!(mcm, V, V_new, U)
    # Simplify notation
    alpha, beta, sigma, c, gamma = mcm.alpha, mcm.beta, mcm.sigma, mcm.c, mcm.gamma

    for (w_idx, w) in enumerate(mcm.w_vec)
        # w_idx indexes the vector of possible wages
        V_new[w_idx] = u(w, sigma) + beta * ((1 - alpha) * V[w_idx] + alpha * U)
    end

    U_new = u(c, sigma) + beta * (1 - gamma) * U +
                    beta * gamma * sum(max(U, V) .* mcm.p_vec)

    return U_new
end


function solve_mccall_model(mcm; tol::Float64=1e-5, max_iter::Int=2000)

    V = ones(length(mcm.w_vec))  # Initial guess of V
    V_new = similar(V)           # To store updates to V
    U = 1.0                        # Initial guess of U
    i = 0
    error = tol + 1

    while error > tol && i < max_iter
        U_new = update_bellman!(mcm, V, V_new, U)
        error_1 = maximum(abs(V_new - V))
        error_2 = abs(U_new - U)
        error = max(error_1, error_2)
        V[:] = V_new
        U = U_new
        i += 1
    end

    return V, U
end
