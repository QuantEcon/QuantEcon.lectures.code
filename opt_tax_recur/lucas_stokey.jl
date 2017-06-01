#=

lucas_stokey.jl

@author: Shunsuke Hori

=#
using QuantEcon
using Dierckx

using NLsolve
using NLopt

type Para{TAF<:AbstractFloat}
    beta::TAF
    Pi::Array{TAF,2}
    G::Vector{TAF}
    Theta::Vector{TAF}
    transfers::Bool
    U::Function
    Uc::Function
    Ucc::Function
    Un::Function
    Unn::Function
end

"""
Class returns planner's allocation as a function of the multiplier
on the implementability constraint mu
"""
type Planners_Allocation_Sequential{TI<:Integer, TAF<:AbstractFloat}
    para::Para{TAF}
    mc::MarkovChain
    S::TI
    cFB::Vector{TAF}
    nFB::Vector{TAF}
    XiFB::Vector{TAF}
    zFB::Vector{TAF}
end

"""
Initializes the class from the calibration Para
"""
function Planners_Allocation_Sequential(para::Para)
    beta, Pi, G, Theta =
        para.beta, para.Pi, para.G, para.Theta
    mc = MarkovChain(Pi)
    S = size(Pi,1) # number of states
    #now find the first best allocation
    cFB, nFB, XiFB, zFB = find_first_best(para,S,1)

    return Planners_Allocation_Sequential(
        para, mc,S, cFB, nFB, XiFB, zFB)
end

"""
Find the first best allocation
"""
function find_first_best(para::Para,S::Integer,
            version::Integer)
    if version != 1 && version != 2
        throw(ArgumentError("version must be 1 or 2"))
    end
    beta, Theta, Uc, Un, G, Pi =
        para.beta, para.Theta, para.Uc, para.Un, para.G, para.Pi
    function res!(z, out)
        c = z[1:S]
        n = z[S+1:end]
        out[1:S] = Theta.*Uc(c,n)+Un(c,n)
        out[S+1:end] = Theta.*n - c - G
    end
    out=zeros(2S)
    res = nlsolve(res!, 0.5*ones(2*S))

    if converged(res) == false
        error("Could not find first best")
    end

    if version == 1
        cFB = res.zero[1:S]
        nFB = res.zero[S+1:end]
        XiFB = Uc(cFB,nFB) #multiplier on the resource constraint.
        zFB = vcat(cFB,nFB,XiFB)
        return cFB, nFB, XiFB, zFB
    elseif version == 2
        cFB = res.zero[1:S]
        nFB = res.zero[S+1:end]
        IFB = Uc(cFB, nFB).*cFB + Un(cFB, nFB).*nFB
        xFB = \(eye(S) - beta*Pi, IFB)
        zFB = Array{Array}(S)
        for s in 1:S
            zFB[s] = vcat(cFB[s], nFB[s], xFB)
        end
        return cFB, nFB, IFB, xFB, zFB
    end
end

"""
Computes optimal allocation for time t\geq 1 for a given \mu
"""
function time1_allocation(pas::Planners_Allocation_Sequential, mu::Real)
    para, S = pas.para, pas.S
    Theta, beta, Pi, G, Uc, Ucc, Un, Unn =
        para.Theta, para.beta, para.Pi, para.G,
        para.Uc, para.Ucc, para.Un, para.Unn
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
    x = \(eye(S) - beta*para.Pi, I)
    return c, n, x, Xi
end

"""
Finds the optimal allocation given initial government debt B_ and state s_0
"""
function time0_allocation(pas::Planners_Allocation_Sequential,
                            B_::AbstractFloat, s_0::Integer)
    para = pas.para
    Pi, Theta, G, beta =
        para.Pi, para.Theta, para.G, para.beta
    Uc, Ucc, Un, Unn =
        para.Uc, para.Ucc, para.Un, para.Unn
    #first order conditions of planner's problem
    function FOC!(z, out)
        mu, c, n, Xi = z[1], z[2], z[3], z[4]
        xprime = time1_allocation(pas, mu)[3]
        out .= vcat(
        Uc(c,n).*(c-B_) + Un(c,n).*n + beta*dot(Pi[s_0,:],xprime),
        Uc(c,n) - mu*(Ucc(c,n).*(c-B_) + Uc(c,n)) - Xi,
        Un(c,n) - mu*(Unn(c,n).*n+Un(c,n)) + Theta[s_0].*Xi,
        (Theta.*n - c - G)[s_0]
        )
    end
    #find root
    res = nlsolve(FOC!,
        [0.0, pas.cFB[s_0], pas.nFB[s_0], pas.XiFB[s_0]])
    if res.f_converged == false
        error("Could not find time 0 LS allocation.")
    end
    return (res.zero...)
end

"""
Find the value associated with multiplier mu
"""
function time1_value(pas::Planners_Allocation_Sequential,mu::Real)
    para= pas.para
    c, n, x, Xi = time1_allocation(pas, mu)
    U_val = para.U(c,n)
    V = \(eye(pas.S) - para.beta*para.Pi, U_val)
    return c, n, x, V
end

"""
Computes Tau given c,n
"""
function Tau(para::Para,c::Union{Real,Vector},n::Union{Real,Vector})
    Uc, Un = para.Uc.(c,n), para.Un.(c,n)
    return 1+Un./(para.Theta .* Uc)
end

"""
Simulates planners policies for T periods
"""
function simulate(pas::Planners_Allocation_Sequential,
                    B_::AbstractFloat, s_0::Integer, T::Integer, sHist=nothing)
    para = pas.para
    Pi, beta, Uc =
        para.Pi, para.beta, para.Uc
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
    mu, cHist[1], nHist[1], _  =
        time0_allocation(pas,B_,s_0)
    TauHist[1] = Tau(pas.para, cHist[1],nHist[1])[s_0]
    Bhist[1] = B_
    muHist[1] = mu
    #time 1 onward
    for t in 2:T
        c, n, x, Xi = time1_allocation(pas,mu)
        u_c = Uc(c,n)
        s = sHist[t]
        TauHist[t] = Tau(pas.para,c,n)[s]
        Eu_c = dot(Pi[sHist[t-1],:],u_c)
        cHist[t], nHist[t], Bhist[t] = c[s], n[s], x[s]/u_c[s]
        RHist[t-1] = Uc(cHist[t-1],nHist[t-1])/(beta*Eu_c)
        muHist[t] = mu
    end
    return cHist, nHist, Bhist, TauHist, sHist, muHist, RHist
end

"""
Bellman equation for the continuation of the Lucas-Stokey Problem
"""
type BellmanEquation{TI<:Integer,TF<:AbstractFloat}
    para::Para
    S::TI
    xbar::Array{TF}
    time_0::Bool
    z0::Array{Array}
    cFB::Vector{TF}
    nFB::Vector{TF}
    xFB::Vector{TF}
    zFB::Vector{Array}
end

"""
Initializes the class from the calibration Para
"""
function BellmanEquation(para::Para,xgrid::AbstractVector,
                        policies0)
    S = size(para.Pi,1) # number of states
    xbar = [minimum(xgrid),maximum(xgrid)]
    time_0 = false
    z0 = Array{Array}(length(xgrid),S)
    cf, nf, xprimef = policies0
    for s in 1:S
        for (i_x, x) in enumerate(xgrid)
            xprime0 = Array{typeof(para.beta)}(S)
            for sprime in 1:S
                xprime0[sprime] = xprimef[s,sprime](x)
            end
            z0[i_x,s] = vcat(cf[s](x),nf[s](x),xprime0)
        end
    end
    cFB, nFB, IFB, xFB, zFB = find_first_best(para, S, 2)
    return BellmanEquation(para,S,xbar,time_0,z0,cFB,nFB,xFB,zFB)
end

"""
Finds the optimal policies
"""
function get_policies_time1(T::BellmanEquation,
                        i_x::Integer, x::AbstractFloat,
                        s::Integer, Vf::Array)
    para, S = T.para, T.S
    beta, Theta, G, Pi = para.beta, para.Theta, para.G, para.Pi
    U, Uc, Un = para.U, para.Uc, para.Un

    function objf(z::Vector,grad)
        c,xprime = z[1], z[2:end]
        n=c+G[s]

        Vprime = Array{typeof(beta)}(S)
        for sprime in 1:S
            Vprime[sprime] = Vf[sprime](xprime[sprime])
        end
        return -(U(c,n)+beta*dot(Pi[s,:],Vprime))
    end
    function cons(z::Vector,grad)
        c, xprime = z[1], z[2:end]
        n=c+G[s]
        return x - Uc(c,n)*c-Un(c,n)*n - beta*dot(Pi[s,:],xprime)
    end
    lb = vcat(0, T.xbar[1]*ones(S))
    ub = vcat(1-G[s], T.xbar[2]*ones(S))
    opt = Opt(:LN_COBYLA, length(T.z0[i_x,s])-1)
    min_objective!(opt, objf)
    equality_constraint!(opt, cons)
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)
    maxeval!(opt, 300)
    maxtime!(opt, 10)
    init = vcat(T.z0[i_x,s][1], T.z0[i_x,s][3:end])
    for (i, val) in enumerate(init)
        if val > ub[i]
            init[i] = ub[i]
        elseif val < lb[i]
            init[i] = lb[i]
        end
    end
    (minf, minx, ret) = optimize(opt, init)
    T.z0[i_x,s] = vcat(minx[1], minx[1]+G[s], minx[2:end])
    return vcat(-minf, T.z0[i_x, s])
end
"""
Finds the optimal policies
"""
function get_policies_time0(T::BellmanEquation,
                        B_::AbstractFloat, s0::Integer, Vf::Array)
    para, S = T.para, T.S
    beta, Theta, G, Pi = para.beta, para.Theta, para.G, para.Pi
    U, Uc, Un = para.U, para.Uc, para.Un
    function objf(z,grad)
        c, xprime = z[1], z[2:end]
        n = c+G[s0]
        Vprime = Array{typeof(beta)}(S)
        for sprime in 1:S
            Vprime[sprime] = Vf[sprime](xprime[sprime])
        end
        return -(U(c,n)+beta*dot(Pi[s0,:],Vprime))
    end
    function cons(z::Vector, grad)
        c, xprime = z[1], z[2:end]
        n = c+G[s0]
        return -Uc(c,n)*(c-B_)-Un(c,n)*n - beta*dot(Pi[s0,:],xprime)
    end
    lb = vcat(0, T.xbar[1]*ones(S))
    ub = vcat(1-G[s0], T.xbar[2]*ones(S))
    opt = Opt(:LN_COBYLA, length(T.zFB[s0])-1)
    min_objective!(opt,objf)
    equality_constraint!(opt,cons)
    lower_bounds!(opt,lb)
    upper_bounds!(opt,ub)
    maxeval!(opt, 300)
    maxtime!(opt, 10)
    init = vcat(T.zFB[s0][1],T.zFB[s0][3:end])
    for (i, val) in enumerate(init)
        if val > ub[i]
            init[i] = ub[i]
        elseif val < lb[i]
            init[i] = lb[i]
        end
    end
    (minf,minx,ret) = optimize(opt,init)
    return vcat(-minf, vcat(minx[1], minx[1]+G[s0], minx[2:end]))
end


"""
Compute the planner's allocation by solving Bellman
equation.
"""
type Planners_Allocation_Bellman
    para::Para
    mc::MarkovChain
    S::Integer
    T::BellmanEquation
    mugrid::Vector
    xgrid::Vector
    Vf::Array
    policies::Array
end

"""
Initializes the class from the calibration Para
"""
function Planners_Allocation_Bellman(para::Para,mugrid::AbstractArray)
    mc = MarkovChain(para.Pi)
    G = para.G
    S = size(para.Pi,1) # number of states
    mugrid = collect(mugrid)
    #now find the first best allocation
    Vf, policies, T, xgrid = solve_time1_bellman(para, mugrid)
    T.time_0 = true #Bellman equation now solves time 0 problem
    return Planners_Allocation_Bellman(
        para,mc,S,T,mugrid,xgrid, Vf,policies)
end

"""
Solve the time 1 Bellman equation for calibration Para and initial grid mugrid0
"""
function solve_time1_bellman(para::Para, mugrid::AbstractArray)
    mugrid0 = mugrid
    S = size(para.Pi,1)
    #First get initial fit
    PP = Planners_Allocation_Sequential(para)
    Tbeta=typeof(PP.para.beta)
    c = Array{Tbeta}(2,length(mugrid))
    n = Array{Tbeta}(2,length(mugrid))
    x = Array{Tbeta}(2,length(mugrid))
    V = Array{Tbeta}(2,length(mugrid))
    for (i,mu) in enumerate(mugrid0)
        c[:,i], n[:,i], x[:,i], V[:,i] = time1_value(PP,mu)
    end
    c=c'
    n=n'
    x=x'
    V=V'
    Vf = Vector{Function}(2)
    cf = Vector{Function}(2)
    nf = Vector{Function}(2)
    xprimef = Array{Function}(2, S)
    for s in 1:2
        splc = Spline1D(vec(x[:,s])[end:-1:1], vec(c[:,s])[end:-1:1], k=3)
        spln = Spline1D(vec(x[:,s])[end:-1:1], vec(n[:,s])[end:-1:1], k=3)
        splV = Spline1D(vec(x[:,s])[end:-1:1], vec(V[:,s])[end:-1:1], k=3)
        cf[s] = x -> evaluate(splc, x)
        nf[s] = x -> evaluate(spln, x)
        Vf[s] = x -> evaluate(splV, x)
        for sprime in 1:S
            splxp = Spline1D(vec(x[:,s])[end:-1:1], vec(x[:,s])[end:-1:1], k=3)
            xprimef[s,sprime] = x -> evaluate(splxp, x)
        end
    end
    policies = [cf,nf,xprimef]
    #create xgrid
    xbar = [maximum(minimum(x,1)),minimum(maximum(x,1))]
    xgrid = linspace(xbar[1],xbar[2],length(mugrid0))
    #Now iterate on bellman equation
    T = BellmanEquation(para, xgrid, policies)
    diff = 1.0
    while diff > 1e-6
        if T.time_0 == false
            Vfnew, policies =
                fit_policy_function(PP,
                (i_x, x, s) -> get_policies_time1(T, i_x, x, s, Vf), xgrid)
        elseif T.time_0 == true
            Vfnew, policies =
                fit_policy_function(PP,
                (i_x, B_, s0) -> get_policies_time0(T, i_x, B_, s0, Vf), xgrid)
        else
            error("T.time_0 is $(T.time_0), which is invalid")
        end
        diff = 0.0
        for s in 1:S
            diff = max(diff,
                    maxabs((Vf[s](xgrid)-Vfnew[s](xgrid))/Vf[s](xgrid))
                        )
        end
        print("diff = $diff \n")
        Vf = Vfnew
    end
    # store value function policies and Bellman Equations
    return Vf, policies, T, xgrid
end

"""
Fits the policy functions PF using the points xgrid using UnivariateSpline
"""
function fit_policy_function(PP::Planners_Allocation_Sequential,
                            PF::Function,xgrid::AbstractArray)
    S = PP.S
    Vf = Vector{Function}(S)
    cf = Vector{Function}(S)
    nf = Vector{Function}(S)
    xprimef = Array{Function}(S, S)
    for s in 1:S
        PFvec = Array{typeof(PP.para.beta)}(length(xgrid), 3+S)
        for (i_x, x) in enumerate(xgrid)
            PFvec[i_x,:] = PF(i_x, x, s)
        end
        splV = Spline1D(xgrid, PFvec[:,1], s=0, k=1)
        splc = Spline1D(xgrid, PFvec[:,2], s=0, k=1)
        spln = Spline1D(xgrid, PFvec[:,3], s=0, k=1)
        Vf[s] = x -> evaluate(splV, x)
        cf[s] = x -> evaluate(splc, x)
        nf[s] = x -> evaluate(spln, x)
        for sprime in 1:S
            splxp = Spline1D(xgrid, PFvec[:,3+sprime], k=1)
            xprimef[s,sprime] = x -> evaluate(splxp, x)
        end
    end
    return Vf, [cf, nf, xprimef]
end

"""
Finds the optimal allocation given initial government debt B_ and state s_0
"""
function time0_allocation(pab::Planners_Allocation_Bellman,
                            B_::AbstractFloat, s0::Integer)
    xgrid = pab.xgrid
    if pab.T.time_0 == false
        z0 = get_policies_time1(pab.T, i_x, x, s, pab.Vf)
    elseif pab.T.time_0 == true
        z0 = get_policies_time0(pab.T, B_, s0, pab.Vf)
    else
        error("T.time_0 is $(T.time_0), which is invalid")
    end
    # PF = T_operator(pab.Vf)
    # z0 = PF(B_, s0)
    c0, n0, xprime0 = z0[2], z0[3], z0[4:end]
    return c0, n0, xprime0
end

"""
Simulates Ramsey plan for T periods
"""
function simulate(pab::Planners_Allocation_Bellman,
                B_::AbstractFloat, s_0::Integer, T::Integer, sHist=nothing)
    para, S, policies = pab.para, pab.S, pab.policies
    beta, Pi, Uc = para.beta, para.Pi, para.Uc
    cf, nf, xprimef = policies[1], policies[2], policies[3]
    if sHist == nothing
        sHist = QuantEcon.simulate(mc, s_0, T)
    end
    cHist = zeros(T)
    nHist = zeros(T)
    Bhist = zeros(T)
    TauHist = zeros(T)
    muHist = zeros(T)
    RHist = zeros(T-1)
    #time0
    cHist[1], nHist[1], xprime = time0_allocation(pab,B_,s_0)
    TauHist[1] = Tau(pab.para,cHist[1], nHist[1])[s_0]
    Bhist[1] = B_
    muHist[1] = 0.0
    #time 1 onward
    for t in 2:T
        s, x = sHist[t], xprime[sHist[t]]
        c, n, xprime = Vector{typeof(beta)}(S), nf[s](x), Vector{typeof(beta)}(S)
        for shat in 1:S
            c[shat] = cf[shat](x)
        end
        for sprime in 1:S
            xprime[sprime] = xprimef[s,sprime](x)
        end
        TauHist[t] = Tau(pab.para,c,n)[s]
        u_c = Uc(c,n)
        Eu_c = dot(Pi[sHist[t-1],:], u_c)
        muHist[t] = pab.Vf[s](x)
        RHist[t-1] = Uc(cHist[t-1],nHist[t-1])/(beta*Eu_c)
        cHist[t], nHist[t], Bhist[t] = c[s], n, x/u_c[s]
    end
    return cHist, nHist, Bhist, TauHist, sHist, muHist, RHist
end
