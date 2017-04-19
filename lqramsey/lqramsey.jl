
#=

This module provides code to compute Ramsey equilibria in a LQ economy with
distortionary taxation.  The program computes allocations (consumption,
leisure), tax rates, revenues, the net present value of the debt and other
related quantities.

Functions for plotting the results are also provided below.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-21

References
----------

Simple port of the file examples/lqramsey.py

http://quant-econ.net/lqramsey.html

=#
using QuantEcon
using PyPlot
#pyplot()
using LaTeXStrings

abstract AbstractStochProcess


type ContStochProcess <: AbstractStochProcess
    A::Matrix
    C::Matrix
end


type DiscreteStochProcess <: AbstractStochProcess
    P::Matrix
    x_vals::Array
end


type Economy{SP <: AbstractStochProcess}
    bet::Real
    Sg::Matrix
    Sd::Matrix
    Sb::Matrix
    Ss::Matrix
    is_discrete::Bool
    proc::SP
end


type Path
    g
    d
    b
    s
    c
    l
    p
    tau
    rvn
    B
    R
    pi
    Pi
    xi
end


function compute_exog_sequences(econ::Economy, x)
    # Compute exogenous variable sequences
    Sg, Sd, Sb, Ss = econ.Sg, econ.Sd, econ.Sb, econ.Ss
    g, d, b, s = [squeeze(S * x, 1) for S in (Sg, Sd, Sb, Ss)]

    #= Solve for Lagrange multiplier in the govt budget constraint
    In fact we solve for nu = lambda / (1 + 2*lambda).  Here nu is the
    solution to a quadratic equation a(nu^2 - nu) + b = 0 where
    a and b are expected discounted sums of quadratic forms of the state. =#
    Sm = Sb - Sd - Ss

    return g, d, b, s, Sm
end


function compute_allocation(econ::Economy, Sm, nu, x, b)
    Sg, Sd, Sb, Ss = econ.Sg, econ.Sd, econ.Sb, econ.Ss

    # Solve for the allocation given nu and x
    Sc = 0.5 .* (Sb + Sd - Sg - nu .* Sm)
    Sl = 0.5 .* (Sb - Sd + Sg - nu .* Sm)
    c = squeeze(Sc * x, 1)
    l = squeeze(Sl * x, 1)
    p = squeeze((Sb - Sc) * x, 1)  # Price without normalization
    tau = 1 .- l ./ (b .- c)
    rvn = l .* tau

    return Sc, Sl, c, l, p, tau, rvn
end


function compute_nu(a0, b0)
    disc = a0^2 - 4a0*b0

    if disc >= 0
        nu = 0.5 *(a0 - sqrt(disc)) / a0
    else
        println("There is no Ramsey equilibrium for these parameters.")
        error("Government spending (economy.g) too low")
    end

    # Test that the Lagrange multiplier has the right sign
    if nu * (0.5 - nu) < 0
        print("Negative multiplier on the government budget constraint.")
        error("Government spending (economy.g) too low")
    end

    return nu
end


function compute_Pi(B, R, rvn, g, xi)
    pi = B[2:end] - R[1:end-1] .* B[1:end-1] - rvn[1:end-1] + g[1:end-1]
    Pi = cumsum(pi .* xi)
    return pi, Pi
end


function compute_paths(econ::Economy{DiscreteStochProcess}, T)
    # simplify notation
    bet, Sg, Sd, Sb, Ss = econ.bet, econ.Sg, econ.Sd, econ.Sb, econ.Ss
    P, x_vals = econ.proc.P, econ.proc.x_vals

    mc = MarkovChain(P)
    state=simulate(mc,T,init=1)
    x = x_vals[:, state]

    # Compute exogenous sequence
    g, d, b, s, Sm = compute_exog_sequences(econ, x)

    # compute a0, b0
    ns = size(P, 1)
    F = eye(ns) - bet.*P
    a0 = (F \ ((Sm * x_vals)'.^2))[1] ./ 2
    H = ((Sb - Sd + Sg) * x_vals) .* ((Sg - Ss)*x_vals)
    b0 = (F \ H')[1] ./ 2

    # compute lagrange multiplier
    nu = compute_nu(a0, b0)

    # Solve for the allocation given nu and x
    Sc, Sl, c, l, p, tau, rvn = compute_allocation(econ, Sm, nu, x, b)

    # compute remaining variables
    H = ((Sb - Sc)*x_vals) .* ((Sl - Sg)*x_vals) - (Sl*x_vals).^2
    temp = squeeze(F*H', 2)
    B = temp[state] ./ p
    H = squeeze(P[state, :] * ((Sb - Sc)*x_vals)', 2)
    R = p ./ (bet .* H)
    temp = squeeze(P[state, :] *((Sb - Sc) * x_vals)', 2)
    xi = p[2:end] ./ temp[1:end-1]

    # compute pi
    pi, Pi = compute_Pi(B, R, rvn, g, xi)

    Path(g, d, b, s, c, l, p, tau, rvn, B, R, pi, Pi, xi)
end


function compute_paths(econ::Economy{ContStochProcess}, T)
    # simplify notation
    bet, Sg, Sd, Sb, Ss = econ.bet, econ.Sg, econ.Sd, econ.Sb, econ.Ss
    A, C = econ.proc.A, econ.proc.C

    # Generate an initial condition x0 satisfying x0 = A x0
    nx, nx = size(A)
    x0 = nullspace((eye(nx) - A))
    x0 = x0[end] < 0 ? -x0 : x0
    x0 = x0 ./ x0[end]
    x0 = squeeze(x0, 2)

    # Generate a time series x of length T starting from x0
    nx, nw = size(C)
    x = zeros(nx, T)
    w = randn(nw, T)
    x[:, 1] = x0
    for t=2:T
        x[:, t] = A *x[:, t-1] + C * w[:, t]
    end

    # Compute exogenous sequence
    g, d, b, s, Sm = compute_exog_sequences(econ, x)

    # compute a0 and b0
    H = Sm'Sm
    a0 = 0.5 * var_quadratic_sum(A, C, H, bet, x0)
    H = (Sb - Sd + Sg)'*(Sg + Ss)
    b0 = 0.5 * var_quadratic_sum(A, C, H, bet, x0)

    # compute lagrange multiplier
    nu = compute_nu(a0, b0)

    # Solve for the allocation given nu and x
    Sc, Sl, c, l, p, tau, rvn = compute_allocation(econ, Sm, nu, x, b)

    # compute remaining variables
    H = Sl'Sl - (Sb - Sc)' *(Sl - Sg)
    L = Array{Float64}(T)
    for t=1:T
        L[t] = var_quadratic_sum(A, C, H, bet, x[:, t])
    end
    B = L ./ p
    Rinv = squeeze(bet .* (Sb- Sc)*A*x, 1) ./ p
    R = 1 ./ Rinv
    AF1 = (Sb - Sc) * x[:, 2:end]
    AF2 = (Sb - Sc) * A * x[:, 1:end-1]
    xi =  AF1 ./ AF2
    xi = squeeze(xi, 1)

    # compute pi
    pi, Pi = compute_Pi(B, R, rvn, g, xi)

    Path(g, d, b, s, c, l, p, tau, rvn, B, R, pi, Pi, xi)
end

function gen_fig_1(path::Path)
    #=
    T = length(path.c)

    tr1, tr2, tr4 = GenericTrace[], GenericTrace[], GenericTrace[]

    # Plot consumption, govt expenditure and revenue
    push!(tr1, scatter(; y = path.rvn, legendgroup = "rev",
     marker_color = "blue", name = L"$\tau_t \ell_t$"))
    push!(tr1, scatter(; y = path.g, legendgroup = "gov",
     marker_color = "red", name = L"$g_t$"))
    push!(tr1, scatter(; y = path.c, marker_color = "green",name = L"$c_t$"))

    # Plot govt expenditure and debt
    push!(tr2, scatter(; x = 1:T, y = path.rvn, legendgroup = "rev",
     marker_color = "blue", showlegend = false, name = L"$\tau_t \ell_t$"))
    push!(tr2, scatter(; x = 1:T, y = path.g, legendgroup = "gov",
     marker_color = "red", showlegend = false, name = L"$g_t$"))
    push!(tr2, scatter(; x = 1:T-1, y = path.B[2:end],
     marker_color = "orange", name = L"$B_{t+1}$"))

    # Plot risk free return
    tr3 = scatter(; x =1:T, y = path.R - 1, marker_color = "pink", name = L"$R_{t - 1}$")

    # Plot revenue, expenditure and risk free rate
    push!(tr4, scatter(; x = 1:T, y = path.rvn, legendgroup = "rev",
     marker_color = "blue",  showlegend = false, name = L"$\tau_t \ell_t$"))
    push!(tr4, scatter(; x = 1:T, y = path.g, legendgroup = "gov", 
     marker_color = "red", showlegend = false, name = L"$g_t$"))
    push!(tr4, scatter(; x = 1:T-1, y = path.pi,
     marker_color = "violet", name = L"$\pi_{t+1}$"))

    p1 = plot(tr1)
    p2 = plot(tr2)
    p3 = plot(tr3, Layout(; xaxis_title = "Time"))
    p4 = plot(tr4, Layout(; xaxis_title = "Time"))
    p = [p1 p2; p3 p4]
    relayout!(p, height = 900)

    p
    =#
    T = length(path.c)
    
    figure(figsize=(12,8))
    
    ax1=subplot(2, 2, 1)
    ax1[:plot](path.rvn)
    ax1[:plot](path.g)
    ax1[:plot](path.c)
    ax1[:set_xlabel]("Time")
    ax1[:legend]([L"$\tau_t \ell_t$",L"$g_t$",L"$c_t$"])
    
    ax2=subplot(2, 2, 2)
    ax2[:plot](path.rvn)
    ax2[:plot](path.g)
    ax2[:plot](path.B[2:end])
    ax2[:set_xlabel]("Time")
    ax2[:legend]([L"$\tau_t \ell_t$",L"$g_t$",L"$B_{t+1}$"])
    
    ax3=subplot(2, 2, 3)
    ax3[:plot](path.R-1)
    ax3[:set_xlabel]("Time")
    ax3[:legend]([L"$R_{t - 1}$"])
    
    ax4=subplot(2, 2, 4)
    ax4[:plot](path.rvn)
    ax4[:plot](path.g)
    ax4[:plot](path.pi)
    ax4[:set_xlabel]("Time")
    ax4[:legend]([L"$\tau_t \ell_t$",L"$g_t$",L"$\pi_{t+1}$"])
end

function gen_fig_2(path::Path)
    #T = length(path.c)

    # Plot adjustment factor
    #p1 = plot(scatter(; x = 2:T, y = path.xi, name = L"$\xi_t$"))

    # Plot adjusted cumulative return
    #p2 = plot(scatter(; x = 2:T, y = path.Pi, name = L"$\Pi_t$"), Layout(; xaxis_title = "Time"))

    #p = [p1; p2]
    #relayout!(p, height = 600)

    #p
    
    T = length(path.c)
    
    figure(figsize=(12,7))
    
    ax1=subplot(2, 1, 1)
    ax1[:plot](2:T, path.xi)
    ax1[:set_xlabel]("Time")
    ax1[:legend]([L"$\xi_t$"])
    
    ax2=subplot(2, 1, 2)
    ax2[:plot](2:T,path.Pi)
    ax2[:set_xlabel]("Time")
    ax2[:legend]([L"$\Pi_t$"])
end
