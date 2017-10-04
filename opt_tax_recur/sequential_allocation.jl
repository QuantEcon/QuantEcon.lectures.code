using QuantEcon
using NLsolve
using NLopt

mutable struct Model{TF <: AbstractFloat,
                     TM <: AbstractMatrix{TF},
                     TV <: AbstractVector{TF}}
    beta::TF
    Pi::TM
    G::TV
    Theta::TV
    transfers::Bool
    U::Function
    Uc::Function
    Ucc::Function
    Un::Function
    Unn::Function
    n_less_than_one::Bool
end

"""
Class returns planner's allocation as a function of the multiplier
on the implementability constraint mu
"""
struct SequentialAllocation{TP <: Model,
                                      TI <: Integer,
                                      TV <: AbstractVector}
    model::TP
    mc::MarkovChain
    S::TI
    cFB::TV
    nFB::TV
    XiFB::TV
    zFB::TV
end

"""
Initializes the class from the calibration model
"""
function SequentialAllocation(model::Model)
    beta, Pi, G, Theta =
        model.beta, model.Pi, model.G, model.Theta
    mc = MarkovChain(Pi)
    S = size(Pi, 1) # number of states
    #now find the first best allocation
    cFB, nFB, XiFB, zFB = find_first_best(model, S, 1)

    return SequentialAllocation(model, mc, S, cFB, nFB, XiFB, zFB)
end

"""
Find the first best allocation
"""
function find_first_best(model::Model, S::Integer, version::Integer)
    if version != 1 && version != 2
        throw(ArgumentError("version must be 1 or 2"))
    end
    beta, Theta, Uc, Un, G, Pi =
        model.beta, model.Theta, model.Uc, model.Un, model.G, model.Pi
    function res!(z, out)
        c = z[1:S]
        n = z[S+1:end]
        out[1:S] = Theta.*Uc(c, n)+Un(c, n)
        out[S+1:end] = Theta.*n - c - G
    end
    res = nlsolve(res!, 0.5*ones(2*S))

    if converged(res) == false
        error("Could not find first best")
    end

    if version == 1
        cFB = res.zero[1:S]
        nFB = res.zero[S+1:end]
        XiFB = Uc(cFB, nFB) #multiplier on the resource constraint.
        zFB = vcat(cFB, nFB, XiFB)
        return cFB, nFB, XiFB, zFB
    elseif version == 2
        cFB = res.zero[1:S]
        nFB = res.zero[S+1:end]
        IFB = Uc(cFB, nFB).*cFB + Un(cFB, nFB).*nFB
        xFB = \(eye(S) - beta*Pi, IFB)
        zFB = [vcat(cFB[s], xFB[s], xFB) for s in 1:S]
        return cFB, nFB, IFB, xFB, zFB
    end
end

"""
Computes optimal allocation for time ``t\geq 1`` for a given ``\mu``
"""
function time1_allocation(pas::SequentialAllocation, mu::Real)
    model, S = pas.model, pas.S
    Theta, beta, Pi, G, Uc, Ucc, Un, Unn =
        model.Theta, model.beta, model.Pi, model.G,
        model.Uc, model.Ucc, model.Un, model.Unn
    function FOC!(z::Vector, out)
        c = z[1:S]
        n = z[S+1:2S]
        Xi = z[2S+1:end]
        out[1:S] = Uc(c,n) - mu*(Ucc(c,n).*c+Uc(c,n)) -Xi #foc c
        out[S+1:2S] = Un(c,n) - mu*(Unn(c,n).*n+Un(c,n)) + Theta.*Xi #foc n
        out[2S+1:end] = Theta.*n - c - G #resource constraint
        return out
    end
    #find the root of the FOC
    res = nlsolve(FOC!, pas.zFB)
    if res.f_converged == false
        error("Could not find LS allocation.")
    end
    z = res.zero
    c, n, Xi = z[1:S], z[S+1:2S], z[2S+1:end]
    #now compute x
    I  = Uc(c,n).*c +  Un(c,n).*n
    x = \(eye(S) - beta*model.Pi, I)
    return c, n, x, Xi
end

"""
Finds the optimal allocation given initial government debt `B_` and state `s_0`
"""
function time0_allocation(pas::SequentialAllocation,
                          B_::AbstractFloat, s_0::Integer)
    model = pas.model
    Pi, Theta, G, beta =
        model.Pi, model.Theta, model.G, model.beta
    Uc, Ucc, Un, Unn =
        model.Uc, model.Ucc, model.Un, model.Unn
    #first order conditions of planner's problem
    function FOC!(z, out)
        mu, c, n, Xi = z[1], z[2], z[3], z[4]
        xprime = time1_allocation(pas, mu)[3]
        out .= vcat(
        Uc(c, n).*(c-B_) + Un(c, n).*n + beta*dot(Pi[s_0, :], xprime),
        Uc(c, n) - mu*(Ucc(c, n).*(c-B_) + Uc(c, n)) - Xi,
        Un(c, n) - mu*(Unn(c, n).*n+Un(c, n)) + Theta[s_0].*Xi,
        (Theta.*n - c - G)[s_0]
        )
    end
    #find root
    res = nlsolve(FOC!, [0.0, pas.cFB[s_0], pas.nFB[s_0], pas.XiFB[s_0]])
    if res.f_converged == false
        error("Could not find time 0 LS allocation.")
    end
    return (res.zero...)
end

"""
Find the value associated with multiplier `mu`
"""
function time1_value(pas::SequentialAllocation, mu::Real)
    model = pas.model
    c, n, x, Xi = time1_allocation(pas, mu)
    U_val = model.U.(c, n)
    V = \(eye(pas.S) - model.beta*model.Pi, U_val)
    return c, n, x, V
end

"""
Computes Tau given `c`, `n`
"""
function Tau(model::Model, c::Union{Real,Vector}, n::Union{Real,Vector})
    Uc, Un = model.Uc.(c, n), model.Un.(c, n)
    return 1+Un./(model.Theta .* Uc)
end

"""
Simulates planners policies for `T` periods
"""
function simulate(pas::SequentialAllocation,
                  B_::AbstractFloat, s_0::Integer, T::Integer,
                  sHist::Union{Vector, Void}=nothing)
    model = pas.model
    Pi, beta, Uc = model.Pi, model.beta, model.Uc
    if sHist == nothing
        sHist = QuantEcon.simulate(pas.mc, T, init=s_0)
    end
    cHist = zeros(T)
    nHist = zeros(T)
    Bhist = zeros(T)
    TauHist = zeros(T)
    muHist = zeros(T)
    RHist = zeros(T-1)
    #time0
    mu, cHist[1], nHist[1], _  = time0_allocation(pas, B_, s_0)
    TauHist[1] = Tau(pas.model, cHist[1], nHist[1])[s_0]
    Bhist[1] = B_
    muHist[1] = mu
    #time 1 onward
    for t in 2:T
        c, n, x, Xi = time1_allocation(pas,mu)
        u_c = Uc(c,n)
        s = sHist[t]
        TauHist[t] = Tau(pas.model, c, n)[s]
        Eu_c = dot(Pi[sHist[t-1],:], u_c)
        cHist[t], nHist[t], Bhist[t] = c[s], n[s], x[s]/u_c[s]
        RHist[t-1] = Uc(cHist[t-1], nHist[t-1])/(beta*Eu_c)
        muHist[t] = mu
    end
    return cHist, nHist, Bhist, TauHist, sHist, muHist, RHist
end

"""
Bellman equation for the continuation of the Lucas-Stokey Problem
"""
mutable struct BellmanEquation{TP <: Model,
                               TI <: Integer,
                               TV <: AbstractVector,
                               TM <: AbstractMatrix{TV},
                               TVV <: AbstractVector{TV}}
    model::TP
    S::TI
    xbar::TV
    time_0::Bool
    z0::TM
    cFB::TV
    nFB::TV
    xFB::TV
    zFB::TVV
end

"""
Initializes the class from the calibration `model`
"""
function BellmanEquation(model::Model, xgrid::AbstractVector, policies0::Vector)
    S = size(model.Pi, 1) # number of states
    xbar = [minimum(xgrid), maximum(xgrid)]
    time_0 = false
    cf, nf, xprimef = policies0
    z0 = [vcat(cf[s](x), nf[s](x), [xprimef[s, sprime](x) for sprime in 1:S])
                        for x in xgrid, s in 1:S]
    cFB, nFB, IFB, xFB, zFB = find_first_best(model, S, 2)
    return BellmanEquation(model, S, xbar, time_0, z0, cFB, nFB, xFB, zFB)
end

"""
Finds the optimal policies
"""
function get_policies_time1(T::BellmanEquation,
                        i_x::Integer, x::AbstractFloat,
                        s::Integer, Vf::AbstractArray)
    model, S = T.model, T.S
    beta, Theta, G, Pi = model.beta, model.Theta, model.G, model.Pi
    U, Uc, Un = model.U, model.Uc, model.Un

    function objf(z::Vector, grad)
        c, xprime = z[1], z[2:end]
        n=c+G[s]
        Vprime = [Vf[sprime](xprime[sprime]) for sprime in 1:S]
        return -(U(c, n) + beta * dot(Pi[s, :], Vprime))
    end
    function cons(z::Vector, grad)
        c, xprime = z[1], z[2:end]
        n=c+G[s]
        return x - Uc(c, n)*c-Un(c, n)*n - beta*dot(Pi[s, :], xprime)
    end
    lb = vcat(0, T.xbar[1]*ones(S))
    ub = vcat(1-G[s], T.xbar[2]*ones(S))
    opt = Opt(:LN_COBYLA, length(T.z0[i_x, s])-1)
    min_objective!(opt, objf)
    equality_constraint!(opt, cons)
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)
    maxeval!(opt, 300)
    maxtime!(opt, 10)
    init = vcat(T.z0[i_x, s][1], T.z0[i_x, s][3:end])
    for (i, val) in enumerate(init)
        if val > ub[i]
            init[i] = ub[i]
        elseif val < lb[i]
            init[i] = lb[i]
        end
    end
    (minf, minx, ret) = optimize(opt, init)
    T.z0[i_x, s] = vcat(minx[1], minx[1]+G[s], minx[2:end])
    return vcat(-minf, T.z0[i_x, s])
end
"""
Finds the optimal policies
"""
function get_policies_time0(T::BellmanEquation,
                        B_::AbstractFloat, s0::Integer, Vf::Array)
    model, S = T.model, T.S
    beta, Theta, G, Pi = model.beta, model.Theta, model.G, model.Pi
    U, Uc, Un = model.U, model.Uc, model.Un
    function objf(z, grad)
        c, xprime = z[1], z[2:end]
        n = c+G[s0]
        Vprime = [Vf[sprime](xprime[sprime]) for sprime in 1:S]
        return -(U(c, n) + beta*dot(Pi[s0, :], Vprime))
    end
    function cons(z::Vector, grad)
        c, xprime = z[1], z[2:end]
        n = c+G[s0]
        return -Uc(c, n)*(c-B_)-Un(c, n)*n - beta*dot(Pi[s0, :], xprime)
    end
    lb = vcat(0, T.xbar[1]*ones(S))
    ub = vcat(1-G[s0], T.xbar[2]*ones(S))
    opt = Opt(:LN_COBYLA, length(T.zFB[s0])-1)
    min_objective!(opt, objf)
    equality_constraint!(opt, cons)
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)
    maxeval!(opt, 300)
    maxtime!(opt, 10)
    init = vcat(T.zFB[s0][1], T.zFB[s0][3:end])
    for (i, val) in enumerate(init)
        if val > ub[i]
            init[i] = ub[i]
        elseif val < lb[i]
            init[i] = lb[i]
        end
    end
    (minf, minx, ret) = optimize(opt, init)
    return vcat(-minf, vcat(minx[1], minx[1]+G[s0], minx[2:end]))
end