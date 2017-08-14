#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#
using QuantEcon
using Roots
using Plots
using LaTeXStrings
pyplot()

struct HistDepRamsey{TF<:AbstractFloat}
    # These are the parameters of the economy
    A0::AbstractFloat
    A1::AbstractFloat
    d::AbstractFloat
    Q0::AbstractFloat
    tau0::AbstractFloat
    mu0::AbstractFloat
    bet::AbstractFloat

    # These are the LQ fields and stationary values
    R::Matrix{TF}
    A::Matrix{TF}
    B::Vector{TF}
    Q::TF
    P::Matrix{TF}
    F::Matrix{TF}
    lq::LQ
end


struct RamseyPath{TF<:AbstractFloat}
    y::Matrix{TF}
    uhat::Vector{TF}
    uhatdif::Vector{TF}
    tauhat::Vector{TF}
    tauhatdif::Vector{TF}
    mu::Vector{TF}
    G::Vector{TF}
    GPay::Vector{TF}
end


function HistDepRamsey(A0::AbstractFloat,
                       A1::AbstractFloat,
                       d::AbstractFloat,
                       Q0::AbstractFloat,
                       tau0::AbstractFloat,
                       mu::AbstractFloat,
                       bet::AbstractFloat)
    # Create Matrices for solving Ramsey problem
    R = [0.0  -A0/2  0.0    0.0
        -A0/2 A1/2   -mu/2  0.0
        0.0   -mu/2  0.0    0.0
        0.0   0.0    0.0    d/2]

    A = [1.0   0.0  0.0  0.0
         0.0   1.0  0.0  1.0
         0.0   0.0  0.0  0.0
         -A0/d A1/d 0.0  A1/d+1.0/bet]

    B = [0.0; 0.0; 1.0; 1.0/d]

    Q = 0.0

    # Use LQ to solve the Ramsey Problem.
    lq = LQ(Q, -R, A, B, bet=bet)

    P, F, _d = stationary_values(lq)

    HistDepRamsey(A0, A1, d, Q0, tau0, mu0, bet, R, A, B, Q, P, Array(F), lq)
end


function compute_G(hdr::HistDepRamsey, mu::AbstractFloat)
    # simplify notation
    Q0, tau0, A, B, Q = hdr.Q0, hdr.tau0, hdr.A, hdr.B, hdr.Q
    bet = hdr.bet

    R = hdr.R
    R[2, 3] = R[3, 2] = -mu/2
    lq = LQ(Q, -R, A, B, bet=bet)

    P, F, _d = stationary_values(lq)

    # Need y_0 to compute government tax revenue.
    u0 = compute_u0(hdr, P)
    y0 = vcat([1.0 Q0 tau0]', u0)

    # Define A_F and S matricies
    AF = A - B * F
    S = [0.0 1.0 0.0 0]' * [0.0 0.0 1.0 0]

    # Solves equation (25)
    Omega = solve_discrete_lyapunov(sqrt(bet) .* AF', bet * AF' * S * AF)
    T0 = y0' * Omega * y0

    return T0[1], A, B, F, P
end


function compute_u0(hdr::HistDepRamsey, P::Matrix)
    # simplify notation
    Q0, tau0 = hdr.Q0, hdr.tau0

    P21 = P[4, 1:3]
    P22 = P[4, 4]
    z0 = [1.0 Q0 tau0]'
    u0 = reshape(-P22^(-1).* P21,1,3)*(z0)

    return u0[1]
end


function init_path{TF<:AbstractFloat}(hdr::HistDepRamsey{TF}, mu0::TF, T::Integer=20)
    # Construct starting values for the path of the Ramsey economy
    G0, A, B, F, P = compute_G(hdr, mu0)

    # Compute the optimal u0
    u0 = compute_u0(hdr, P)

    # Initialize vectors
    y = Array{TF}(4, T)
    uhat       = Vector{TF}(T)
    uhatdif    = Vector{TF}(T)
    tauhat     = Vector{TF}(T)
    tauhatdif  = Vector{TF}(T)
    mu         = Vector{TF}(T)
    G          = Vector{TF}(T)
    GPay       = Vector{TF}(T)

    # Initial conditions
    G[1] = G0
    mu[1] = mu0
    uhatdif[1] = 0
    uhat[1] = u0
    tauhatdif[1] = 0
    y[:, 1] = vcat([1.0 hdr.Q0 hdr.tau0]', u0)

    return RamseyPath(y, uhat, uhatdif, tauhat, tauhatdif, mu, G, GPay)
end


function compute_ramsey_path!(hdr::HistDepRamsey, rp::RamseyPath)
    # simplify notation
    y, uhat, uhatdif, tauhat, = rp.y, rp.uhat, rp.uhatdif, rp.tauhat
    tauhatdif, mu, G, GPay = rp.tauhatdif, rp.mu, rp.G, rp.GPay
    bet = hdr.bet

    _, A, B, F, P = compute_G(hdr, mu[1])


    for t=2:T
        # iterate government policy
        y[:, t] = (A - B * F) * y[:, t-1]

        # update G
        G[t] = (G[t-1] - bet*y[2, t]*y[3, t])/bet
        GPay[t] = bet.*y[2, t]*y[3, t]

        #=
        Compute the mu if the government were able to reset its plan
        ff is the tax revenues the government would receive if they reset the
        plan with Lagrange multiplier mu minus current G
        =#
        ff(mu) = compute_G(hdr, mu)[1]-G[t]

        # find ff = 0
        mu[t] = fzero(ff, mu[t-1])
        _, Atemp, Btemp, Ftemp, Ptemp = compute_G(hdr, mu[t])

        # Compute alternative decisions
        P21temp = Ptemp[4, 1:3]
        P22temp = P[4, 4]
        uhat[t] = dot(-P22temp^(-1) .* P21temp, y[1:3, t])

        yhat = (Atemp-Btemp * Ftemp) * [y[1:3, t-1]; uhat[t-1]]
        tauhat[t] = yhat[4]
        tauhatdif[t] = tauhat[t] - y[4, t]
        uhatdif[t] = uhat[t] - y[4, t]
    end

    return rp
end


function plot1(rp::RamseyPath)
    tt = 0:length(rp.mu)  # tt is used to make the plot time index correct.
    y = rp.y

    ylabels = [L"$Q$" L"$\tau$" L"$u$"]
    #y_vals = [squeeze(y[2, :], 1) squeeze(y[3, :], 1) squeeze(y[4, :], 1)]
    y_vals = [y[2, :] y[3, :] y[4, :]]
    p = plot(tt, y_vals, color=:blue,
            label=["output" "tax rate" "first difference in output"],
            lw=2, alpha=0.7, ylabel=ylabels, layout=(3,1),
            xlims=(0, 15), xlabel=["" "" "time"], legend=:topright,
            xticks=0:5:15)
     return p
end

function plot2(rp::RamseyPath)
    y, uhatdif, tauhatdif, mu = rp.y, rp.uhatdif, rp.tauhatdif, rp.mu
    G, GPay = rp.G, rp.GPay
    T = length(rp.mu)
    tt = 0:T  # tt is used to make the plot time index correct.

    y_vals = [tauhatdif uhatdif mu G]
    ylabels = [L"$\Delta\tau$" L"$\Delta u$" L"$\mu$" L"$G$"]
    labels = hcat("time inconsistency differential for tax rate",
              L"time inconsistency differential for $u$",
              "Lagrange multiplier", "government revenue")
    p = plot(tt, y_vals, ylabel=ylabels, label=labels,
             layout=(4, 1), xlims=(-0.5, 15), lw=2, alpha=0.7,
             legend=:topright, color=:blue, xlabel=["" "" "" "time"])
    return p
end

# Primitives
T    = 20
A0   = 100.0
A1   = 0.05
d    = 0.20
bet = 0.95

# Initial conditions
mu0  = 0.0025
Q0   = 1000.0
tau0 = 0.0

# Solve Ramsey problem and compute path
hdr = HistDepRamsey(A0, A1, d, Q0, tau0, mu0, bet)
rp = init_path(hdr, mu0, T)
compute_ramsey_path!(hdr, rp)  # updates rp in place
p1=plot1(rp)
display(p1)
p2=plot2(rp)
display(p2)
