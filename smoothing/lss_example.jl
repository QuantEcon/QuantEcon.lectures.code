"""
Computes the path of consumption and debt for the previously described
complete markets model where exogenous income follows a linear
state space
"""
function complete_ss(beta::AbstractFloat,
                     b0::Union{AbstractFloat, Array},
                     x0::Union{AbstractFloat, Array},
                     A::Union{AbstractFloat, Array},
                     C::Union{AbstractFloat, Array},
                     S_y::Union{AbstractFloat, Array},
                     T::Integer=12)

    # Create a linear state space for simulation purposes
    # This adds "b" as a state to the linear state space system
    # so that setting the seed places shocks in same place for
    # both the complete and incomplete markets economy
    # Atilde = vcat(hcat(A, zeros(size(A,1), 1)),
    #               zeros(1, size(A,2) + 1))
    # Ctilde = vcat(C, zeros(1, 1))
    # S_ytilde = hcat(S_y, zeros(1, 1))

    lss = LSS(A, C, S_y, mu_0=x0)

    # Add extra state to initial condition
    # x0 = hcat(x0, 0)

    # Compute the (I - beta*A)^{-1}
    rm = inv(eye(size(A,1)) - beta*A)

    # Constant level of consumption
    cbar = (1-beta) * (S_y * rm * x0 - b0)
    c_hist = ones(T)*cbar[1]

    # Debt
    x_hist, y_hist = simulate(lss, T)
    b_hist = (S_y * rm * x_hist - cbar[1]/(1.0-beta))


    return c_hist, vec(b_hist), vec(y_hist), x_hist
end

using PyPlot

N_simul = 150

# Define parameters
alpha, rho1, rho2 = 10.0, 0.9, 0.0
sigma = 1.0
# N_simul = 1
# T = N_simul
A = [1.0 0.0 0.0;
     alpha rho1 rho2;
     0.0 1.0 0.0]
C = [0.0, sigma, 0.0]
S_y = [1.0 1.0 0.0]
beta, b0 = 0.95, -10.0
x0 = [1.0, alpha/(1-rho1), alpha/(1-rho1)]

# Do simulation for complete markets
s = rand(1:10000)
srand(s)  # Seeds get set the same for both economies
out = complete_ss(beta, b0, x0, A, C, S_y, 150)
c_hist_com, b_hist_com, y_hist_com, x_hist_com = out


fig, ax = subplots(1, 2, figsize = (15, 5))

# Consumption plots
ax[1][:set_title]("Cons and income", fontsize = 17)
ax[1][:plot](1:N_simul, c_hist_com, label = "consumption", lw = 3)
ax[1][:plot](1:N_simul, y_hist_com, label = "income",
            lw = 2, alpha = 0.6, linestyle = "--")
ax[1][:legend](loc = "best", fontsize = 15)
ax[1][:set_xlabel]("Periods", fontsize = 13)
ax[1][:set_ylim]([-5.0, 110])

    # Debt plots
ax[2][:set_title]("Debt and income", fontsize = 17)
ax[2][:plot](1:N_simul, b_hist_com, label = "debt", lw = 2)
ax[2][:plot](1:N_simul, y_hist_com, label = "Income",
            lw = 2, alpha = 0.6, linestyle = "--")
ax[2][:legend](loc = "best", fontsize = 15)
ax[2][:axhline](0, color = "k", lw = 1)
ax[2][:set_xlabel]("Periods", fontsize = 13)
