"""
This type is used to store the solution to the problem presented
in the "Wald Friedman" notebook presented on the QuantEcon website.

Solution
----------
J : vector(Float64)
    Discretized value function that solves the Bellman equation
lb : scalar(Real)
    Lower cutoff for continuation decision
ub : scalar(Real)
    Upper cutoff for continuation decision
"""
type WFSolution
    J::Vector{Float64}
    lb::Real
    ub::Real
end

"""
This type is used to solve the problem presented in the "Wald Friedman"
notebook presented on the QuantEcon website.

Parameters
----------
c : scalar(Real)
    Cost of postponing decision
L0 : scalar(Real)
    Cost of choosing model 0 when the truth is model 1
L1 : scalar(Real)
    Cost of choosing model 1 when the truth is model 0
f0 : vector(Float64)
    A finite state probability distribution
f1 : vector(Float64)
    A finite state probability distribution
m : scalar(Int64)
    Number of points to use in function approximation
"""
immutable WaldFriedman
    c::Real
    L0::Real
    L1::Real
    f0::Vector{Float64}
    f1::Vector{Float64}
    m::Int64
    pgrid::LinSpace{Float64}
    sol::WFSolution
end

function WaldFriedman(c, L0, L1, f0, f1; m=25)
    pgrid = linspace(0.0, 1.0, m)

    # Renormalize distributions so nothing is "too" small
    f0 = clamp(f0, 1e-8, 1-1e-8)
    f1 = clamp(f1, 1e-8, 1-1e-8)
    f0 = f0 / sum(f0)
    f1 = f1 / sum(f1)
    J = zeros(m)
    lb = 0.
    ub = 0.

    WaldFriedman(c, L0, L1, f0, f1, m, pgrid, WFSolution(J, lb, ub))
end

"""
This function takes a value for the probability with which
the correct model is model 0 and returns the mixed
distribution that corresponds with that belief.
"""
function current_distribution(wf::WaldFriedman, p)
    return p*wf.f0 + (1-p)*wf.f1
end

"""
This function takes a value for p, and a realization of the
random variable and calculates the value for p tomorrow.
"""
function bayes_update_k(wf::WaldFriedman, p, k)
    f0_k = wf.f0[k]
    f1_k = wf.f1[k]

    p_tp1 = p*f0_k / (p*f0_k + (1-p)*f1_k)

    return clamp(p_tp1, 0, 1)
end

"""
This is similar to `bayes_update_k` except it returns a
new value for p for each realization of the random variable
"""
function bayes_update_all(wf::WaldFriedman, p)
    return clamp(p*wf.f0 ./ (p*wf.f0 + (1-p)*wf.f1), 0, 1)
end

"""
For a given probability specify the cost of accepting model 0
"""
function payoff_choose_f0(wf::WaldFriedman, p)
    return (1-p)*wf.L0
end

"""
For a given probability specify the cost of accepting model 1
"""
function payoff_choose_f1(wf::WaldFriedman, p)
    return p*wf.L1
end

"""
This function evaluates the expectation of the value function
at period t+1. It does so by taking the current probability
distribution over outcomes:

    p(z_{k+1}) = p_k f_0(z_{k+1}) + (1-p_k) f_1(z_{k+1})

and evaluating the value function at the possible states
tomorrow J(p_{t+1}) where

    p_{t+1} = p f0 / ( p f0 + (1-p) f1)

Parameters
----------
p : scalar
    The current believed probability that model 0 is the true
    model.
J : function (interpolant)
    The current value function for a decision to continue

Returns
-------
EJ : scalar
    The expected value of the value function tomorrow
"""
function EJ(wf::WaldFriedman, p, J)
    # Pull out information
    f0, f1 = wf.f0, wf.f1

    # Get the current believed distribution and tomorrows possible dists
    # Need to clip to make sure things don't blow up (go to infinity)
    curr_dist = current_distribution(wf, p)
    tp1_dist = bayes_update_all(wf, p)

    # Evaluate the expectation
    EJ = dot(curr_dist, J.(tp1_dist))

    return EJ
end

"""
For a given probability distribution and value function give
cost of continuing the search for correct model
"""
function payoff_continue(wf::WaldFriedman, p, J)
    return wf.c + EJ(wf, p, J)
end

"""
Evaluates the value function for a given continuation value
function; that is, evaluates

    J(p) = min( (1-p)L0, pL1, c + E[J(p')])

Uses linear interpolation between points
"""
function bellman_operator(wf::WaldFriedman, J)
    c, L0, L1, f0, f1 = wf.c, wf.L0, wf.L1, wf.f0, wf.f1
    m, pgrid = wf.m, wf.pgrid

    J_out = zeros(m)
    J_interp = LinInterp(pgrid, J)

    for (p_ind, p) in enumerate(pgrid)
        # Payoff of choosing model 0
        p_c_0 = payoff_choose_f0(wf, p)
        p_c_1 = payoff_choose_f1(wf, p)
        p_con = payoff_continue(wf, p, J_interp)

        J_out[p_ind] = min(p_c_0, p_c_1, p_con)
    end

    return J_out
end

"""
This function takes a value function and returns the corresponding
cutoffs of where you transition between continue and choosing a
specific model
"""
function find_cutoff_rule(wf::WaldFriedman, J)
    m, pgrid = wf.m, wf.pgrid

    # Evaluate cost at all points on grid for choosing a model
    p_c_0 = payoff_choose_f0(wf, pgrid)
    p_c_1 = payoff_choose_f1(wf, pgrid)

    # The cutoff points can be found by differencing these costs with
    # the Bellman equation (J is always less than or equal to p_c_i)
    lb = pgrid[searchsortedlast(p_c_1 - J, 1e-10)]
    ub = pgrid[searchsortedlast(J - p_c_0, -1e-10)]

    return lb, ub
end

function solve_model(wf; tol=1e-7)
    bell_op(x) = bellman_operator(wf, x)
    J =  compute_fixed_point(bell_op, zeros(wf.m), err_tol=tol, print_skip=5)

    wf.sol.J = J
    wf.sol.lb, wf.sol.ub = find_cutoff_rule(wf, J)
    return J
end

"""
This function takes an initial condition and simulates until it
stops (when a decision is made).
"""
function simulate(wf::WaldFriedman, f; p0=0.5)
    # Check whether vf is computed
    if sumabs(wf.sol.J) < 1e-8
        solve_model(wf)
    end

    # Unpack useful info
    lb, ub = wf.sol.lb, wf.sol.ub
    drv = DiscreteRV(f)

    # Initialize a couple useful variables
    decision_made = false
    decision = 0
    p = p0
    t = 0

    while !decision_made
        # Maybe should specify which distribution is correct one so that
        # the draws come from the "right" distribution
        k = rand(drv)
        t = t+1
        p = bayes_update_k(wf, p, k)
        if p < lb
            decision_made = true
            decision = 1
        elseif p > ub
            decision_made = true
            decision = 0
        end
    end

    return decision, p, t
end

"""
Uses the distribution f0 as the true data generating
process
"""
function simulate_tdgp_f0(wf::WaldFriedman; p0=0.5)
    decision, p, t = simulate(wf, wf.f0; p0=p0)

    if decision == 0
        correct = true
    else
        correct = false
    end

    return correct, p, t
end

"""
Uses the distribution f1 as the true data generating
process
"""
function simulate_tdgp_f1(wf::WaldFriedman; p0=0.5)
    decision, p, t = simulate(wf, wf.f1; p0=p0)

    if decision == 1
        correct = true
    else
        correct = false
    end

    return correct, p, t
end

"""
Simulates repeatedly to get distributions of time needed to make a
decision and how often they are correct.
"""
function stopping_dist(wf::WaldFriedman; ndraws=250, tdgp="f0")
    if tdgp=="f0"
        simfunc = simulate_tdgp_f0
    else
        simfunc = simulate_tdgp_f1
    end

    # Allocate space
    tdist = Array(Int64, ndraws)
    cdist = Array(Bool, ndraws)

    for i in 1:ndraws
        correct, p, t = simfunc(wf)
        tdist[i] = t
        cdist[i] = correct
    end

    return cdist, tdist
end
