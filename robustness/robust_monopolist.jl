
#=

The robust control problem for a monopolist with adjustment costs.  The
inverse demand curve is:

  p_t = a_0 - a_1 y_t + d_t

where d_{t+1} = \rho d_t + \sigma_d w_{t+1} for w_t ~ N(0,1) and iid.
The period return function for the monopolist is

  r_t =  p_t y_t - gam (y_{t+1} - y_t)^2 / 2 - c y_t

The objective of the firm is E_t \sum_{t=0}^\infty \beta^t r_t

For the linear regulator, we take the state and control to be

    x_t = (1, y_t, d_t) and u_t = y_{t+1} - y_t

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-07-05

References
----------

Simple port of the file examples/robust_monopolist.py

http://quant-econ.net/robustness.html#application

=#
using QuantEcon
using Plots
pyplot()

# model parameters
a_0     = 100
a_1     = 0.5
rho     = 0.9
sigma_d = 0.05
bet    = 0.95
c       = 2
gam   = 50.0
theta = 0.002
ac    = (a_0 - c) / 2.0

# Define LQ matrices
R = [0 ac    0;
     ac -a_1 0.5;
     0. 0.5  0]
R = -R  # For minimization
Q = Matrix([gam / 2.0]')
A = [1. 0. 0.;
     0. 1. 0.;
     0. 0. rho]
B = [0. 1. 0.]'
C = [0. 0. sigma_d]'

## Functions

function evaluate_policy(theta::AbstractFloat, F::AbstractArray)
    rlq = RBLQ(Q, R, A, B, C, bet, theta)
    K_F, P_F, d_F, O_F, o_F = evaluate_F(rlq, F)
    x0 = [1.0 0.0 0.0]'
    value = - x0'*P_F*x0 - d_F
    entropy = x0'*O_F*x0 + o_F
    return value[1], entropy[1]  # return scalars
end


function value_and_entropy{TF<:AbstractFloat}(emax::AbstractFloat,
                                              F::AbstractArray{TF},
                                              bw::String,
                                              grid_size::Integer=1000)
    if lowercase(bw) == "worst"
        thetas = 1 ./ linspace(1e-8, 1000, grid_size)
    else
        thetas = -1 ./ linspace(1e-8, 1000, grid_size)
    end

    data = Array{TF}(grid_size, 2)

    for (i, theta) in enumerate(thetas)
        data[i, :] = collect(evaluate_policy(theta, F))
        if data[i, 2] >= emax  # stop at this entropy level
            data = data[1:i, :]
            break
        end
    end
    return data
end

## Main

# compute optimal rule
optimal_lq = LQ(Q, R, A, B, C, zero(B'A), bet=bet)
Po, Fo, Do = stationary_values(optimal_lq)

# compute robust rule for our theta
baseline_robust = RBLQ(Q, R, A, B, C, bet, theta)
Fb, Kb, Pb = robust_rule(baseline_robust)

# Check the positive definiteness of worst-case covariance matrix to
# ensure that theta exceeds the breakdown point
test_matrix = eye(size(Pb, 1)) - (C' * Pb * C ./ theta)[1]
eigenvals, eigenvecs = eig(test_matrix)
@assert all(eigenvals .>= 0)

emax = 1.6e6

# compute values and entropies
optimal_best_case = value_and_entropy(emax, Fo, "best")
robust_best_case = value_and_entropy(emax, Fb, "best")
optimal_worst_case = value_and_entropy(emax, Fo, "worst")
robust_worst_case = value_and_entropy(emax, Fb, "worst")

# we reverse order of "worst_case"s so values are ascending
data_pairs = ((optimal_best_case, optimal_worst_case),
              (robust_best_case, robust_worst_case))

egrid = linspace(0, emax, 100)
egrid_data = Array{Float64}[]
for data_pair in data_pairs
    for data in data_pair
        x, y = data[:, 2], data[:, 1]
        curve = LinInterp(x, y)
        push!(egrid_data, curve.(egrid))
    end
end
plot(egrid, egrid_data, color=[:red :red :blue :blue])
plot!(egrid, egrid_data[1], fillrange=egrid_data[2],
      fillcolor=:red, fillalpha=0.1, color=:red, legend=:none)
plot!(egrid, egrid_data[3], fillrange=egrid_data[4],
      fillcolor=:blue, fillalpha=0.1, color=:blue, legend=:none)
plot!(xlabel="Entropy", ylabel="Value")
