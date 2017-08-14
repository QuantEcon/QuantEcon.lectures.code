#=

Author: Shunsuke Hori

=#

include("lucas_stokey.jl")
using LS

using QuantEcon
using NLopt
using NLsolve
using Dierckx

Para=LS.Para
Planners_Allocation_Sequential = LS.Planners_Allocation_Sequential
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
    policies
end

"""
Initializes the type from the calibration Para
"""
function Planners_Allocation_Bellman(para::Para,mugrid::AbstractArray)
    G = para.G
    S = size(para.Pi,1) # number of states
    mc=MarkovChain(para.Pi)
    mugrid = collect(mugrid)
    #now find the first best allocation
    Vf, policies, T, xgrid = solve_time1_bellman(para, mugrid)
    T.time_0 = true #Bellman equation now solves time 0 problem
    return Planners_Allocation_Bellman(para,mc,S,T,mugrid,xgrid, Vf, policies)
end

"""
Solve the time  1 Bellman equation for calibration Para and initial grid mugrid
"""
function solve_time1_bellman{TR<:Real}(para::Para,mugrid::AbstractArray{TR})
    Pi = para.Pi
    S = size(para.Pi,1)

    #First get initial fit from lucas stockey solution.
    #Need to change things to be ex_ante
    PP = LS.Planners_Allocation_Sequential(para)

    function incomplete_allocation(PP::LS.Planners_Allocation_Sequential,
                                    mu_::AbstractFloat, s_::Integer)
        c,n,x,V = LS.time1_value(PP,mu_)
        return c,n,dot(Pi[s_,:], x), dot(Pi[s_,:], V)
    end
    cf = Array{Function}(S,S)
    nf = Array{Function}(S,S)
    xprimef = Array{Function}(S,S)
    Vf = Array{Function}(S)
    # TR=Float64
    xgrid = Array{TR}(S,length(mugrid))
    for s_ in 1:S
        c = Array{TR}(length(mugrid),S)
        n = Array{TR}(length(mugrid),S)
        x = Array{TR}(length(mugrid))
        V = Array{TR}(length(mugrid))
        for (i_mu, mu) in enumerate(mugrid)
            c[i_mu,:], n[i_mu,:], x[i_mu], V[i_mu] =
                incomplete_allocation(PP, mu, s_)
        end
        xprimes = repmat(x,1,S)
        xgrid[s_,:] = x
        for sprime = 1:S
            splc = Spline1D(x[end:-1:1],c[:,sprime][end:-1:1],k=3)
            spln = Spline1D(x[end:-1:1],n[:,sprime][end:-1:1],k=3)
            splx = Spline1D(x[end:-1:1],xprimes[:,sprime][end:-1:1],k=3)
            cf[s_,sprime] = y -> splc(y)
            nf[s_,sprime] = y -> spln(y)
            xprimef[s_,sprime] = y -> splx(y)
        end
        splV = Spline1D(x[end:-1:1],V[end:-1:1],k=3)
        Vf[s_] = y -> splV(y)
    end

    policies = [cf,nf,xprimef]

    #create xgrid
    xbar = [maximum(minimum(xgrid)), minimum(maximum(xgrid))]
    xgrid = collect(linspace(xbar[1],xbar[2],length(mugrid)))

    #Now iterate on Bellman equation
    T = BellmanEquation(para,xgrid,policies)
    diff = 1.0
    while diff > 1e-6
        PF = (i_x, x,s) -> get_policies_time1(T, i_x, x, s, Vf, xbar)
        Vfnew, policies = fit_policy_function(T, PF, xgrid)

        diff = 0.0
        for s=1:S
            diff = max(diff,maxabs((Vf[s].(xgrid)-Vfnew[s].(xgrid))./Vf[s].(xgrid)))
        end

        println("diff = $diff")
        Vf = copy(Vfnew)
    end

    return Vf, policies, T, xgrid
end

"""
Fits the policy functions
"""
function fit_policy_function{TF<:AbstractFloat}(
                    T::BellmanEquation,
                    PF::Function, xgrid::AbstractArray{TF})
    S = T.S
    # preallocation
    # TF = Float64
    PFvec = Array{TF}(4S+1,length(xgrid))
    cf = Array{Function}(S,S)
    nf = Array{Function}(S,S)
    xprimef = Array{Function}(S,S)
    TTf = Array{Function}(S,S)
    Vf = Array{Function}(S)
    # fit policy fuctions
    for s_ in 1:S
        for (i_x, x) in enumerate(xgrid)
            PFvec[:,i_x] = PF(i_x,x,s_)
        end
        splV = Spline1D(xgrid, PFvec[1,:], k=3)
        Vf[s_] = y -> splV(y)
        for sprime=1:S
            splc = Spline1D(xgrid, PFvec[1+sprime,:], k=3)
            spln = Spline1D(xgrid, PFvec[1+S+sprime,:], k=3)
            splxprime = Spline1D(xgrid, PFvec[1+2S+sprime,:], k=3)
            splTT = Spline1D(xgrid, PFvec[1+3S+sprime,:], k=3)
            cf[s_,sprime] = y -> splc(y)
            nf[s_,sprime] = y -> spln(y)
            xprimef[s_,sprime] = y -> splxprime(y)
            TTf[s_,sprime] = y -> splTT(y)
        end
    end
    policies = (cf, nf, xprimef, TTf)
    return Vf, policies
end

"""
Computes Tau given c,n
"""
function Tau(pab::Planners_Allocation_Bellman,
            c::Array,n::Array)
    para = pab.para
    Uc, Un = para.Uc(c,n), para.Un(c,n)
    return 1+Un./(para.Theta .* Uc)
end

Tau(pab::Planners_Allocation_Bellman, c::AbstractFloat,n::AbstractFloat) =
    Tau(pab, [c], [n])

"""
Finds the optimal allocation given initial government debt B_ and state s_0
"""
function time0_allocation(pab::Planners_Allocation_Bellman,
                           B_::Real,s0::Integer)
    T, Vf = pab.T, pab.Vf
    xbar = T.xbar
    z0 = get_policies_time0(T, B_, s0, Vf, xbar)

    c0, n0, xprime0, T0 = z0[2], z0[3], z0[4], z0[5]
    return c0, n0, xprime0, T0
end

"""
Simulates planners policies for T periods
"""
function simulate{TI<:Integer,TF<:AbstractFloat}(pab::Planners_Allocation_Bellman,
                   B_::TF,s_0::TI,T::TI,sHist=nothing)
    para, mc, Vf, S = pab.para, pab.mc, pab.Vf, pab.S
    Pi, Uc = para.Pi, para.Uc
    cf,nf,xprimef,TTf = pab.policies

    if sHist == nothing
        sHist = QuantEcon.simulate(mc,T,init=s_0)
    end

    cHist=Array{TF}(T)
    nHist=Array{TF}(T)
    Bhist=Array{TF}(T)
    xHist=Array{TF}(T)
    TauHist=Array{TF}(T)
    THist=Array{TF}(T)
    muHist=Array{TF}(T)

    #time0
    cHist[1],nHist[1],xHist[1],THist[1]  =
        time0_allocation(pab, B_,s_0)
    TauHist[1] = Tau(pab,cHist[1],nHist[1])[s_0]
    Bhist[1] = B_
    muHist[1] = Vf[s_0](xHist[1])

    #time 1 onward
    for t in 2:T
        s_, x, s = sHist[t-1], xHist[t-1], sHist[t]
        c = Array{TF}(S)
        n = Array{TF}(S)
        xprime = Array{TF}(S)
        TT = Array{TF}(S)
        for sprime=1:S
            c[sprime], n[sprime], xprime[sprime], TT[sprime] =
                cf[s_,sprime](x), nf[s_,sprime](x),
                xprimef[s_,sprime](x), TTf[s_,sprime](x)
        end

        Tau_val = Tau(pab, c, n)[s]
        u_c = Uc(c,n)
        Eu_c = dot(Pi[s_,:], u_c)

        muHist[t] = Vf[s](xprime[s])

        cHist[t], nHist[t], Bhist[t], TauHist[t] = c[s], n[s], x/Eu_c, Tau_val
        xHist[t],THist[t] = xprime[s],TT[s]
    end
    return cHist,nHist,Bhist,xHist,TauHist,THist,muHist,sHist
end

"""
Initializes the class from the calibration Para
"""
function BellmanEquation{TF<:AbstractFloat}(para::Para,
        xgrid::AbstractVector{TF}, policies0::Array)
    S = size(para.Pi,1) # number of states
    xbar = [minimum(xgrid),maximum(xgrid)]
    time_0 = false
    z0 = Array{Array}(length(xgrid),S)
    cf, nf, xprimef =
        policies0[1], policies0[2], policies0[3]
    for s in 1:S
        for (i_x, x) in enumerate(xgrid)
            cs=Array{TF}(S)
            ns=Array{TF}(S)
            xprimes=Array{TF}(S)
            for j=1:S
                cs[j], ns[j], xprimes[j] =
                    cf[s,j](x), nf[s,j](x), xprimef[s,j](x)
            end
            z0[i_x,s] =
                vcat(cs,ns,xprimes,zeros(S))
        end
    end
    cFB, nFB, IFB, xFB, zFB = LS.find_first_best(para, S, 2)
    return BellmanEquation(para,S,xbar,time_0,z0,cFB,nFB,xFB,zFB)
end


"""
Finds the optimal policies
"""
function get_policies_time1{TF<:AbstractFloat}(T::BellmanEquation,
        i_x::Integer, x::TF, s_::Integer, Vf::Array{Function}, xbar::Vector)
    para, S = T.para, T.S
    beta, Theta, G, Pi = para.beta, para.Theta, para.G, para.Pi
    U,Uc,Un = para.U, para.Uc, para.Un

    S_possible = sum(Pi[s_,:].>0)
    sprimei_possible = find(Pi[s_,:].>0)

    function objf(z,grad)
        c, xprime = z[1:S_possible], z[S_possible+1:2S_possible]
        n=(c+G[sprimei_possible])./Theta[sprimei_possible]
        Vprime = Vector{TF}(S_possible)
        for (si, s) in enumerate(sprimei_possible)
            Vprime[si] = Vf[s](xprime[si])
        end
        return -dot(Pi[s_,sprimei_possible], U(c,n)+beta*Vprime)
    end

    function cons(out,z,grad)
        c, xprime, TT =
            z[1:S_possible], z[S_possible+1:2S_possible], z[2S_possible+1:3S_possible]
        n=(c+G[sprimei_possible])./Theta[sprimei_possible]
        u_c = Uc(c,n)
        Eu_c = dot(Pi[s_,sprimei_possible], u_c)
        out .= x*u_c/Eu_c - u_c.*(c-TT)-Un(c,n).*n - beta*xprime
    end
    function cons_no_trans(out,z,grad)
        c, xprime =
            z[1:S_possible], z[S_possible+1:2S_possible]
        n=(c+G[sprimei_possible])./Theta[sprimei_possible]
        u_c = Uc(c,n)
        Eu_c = dot(Pi[s_,sprimei_possible], u_c)
        out .= x*u_c/Eu_c - u_c.*c-Un(c,n).*n - beta*xprime
    end

    if para.transfers == true
        lb = vcat(zeros(S_possible), ones(S_possible)*xbar[1], zeros(S_possible))
        if para.n_less_than_one == true
            ub = vcat(100*ones(S_possible)-G[sprimei_possible], ones(S_possible)*xbar[2], 100*ones(S_possible))
        else
            ub = vcat(100*ones(S_possible), ones(S_possible)*xbar[2], 100*ones(S_possible))
        end
            init = vcat(T.z0[i_x,s_][sprimei_possible],
                T.z0[i_x,s_][2S+sprimei_possible], T.z0[i_x,s_][3S+sprimei_possible])
        opt = Opt(:LN_COBYLA, 3S_possible)
        equality_constraint!(opt, cons, zeros(S_possible))
    else
        lb = vcat(zeros(S_possible), ones(S_possible)*xbar[1])
        if para.n_less_than_one == true
            ub = vcat(ones(S_possible)-G[sprimei_possible], ones(S_possible)*xbar[2])
        else
            ub = vcat(ones(S_possible), ones(S_possible)*xbar[2])
        end
        init = vcat(T.z0[i_x,s_][sprimei_possible],
                    T.z0[i_x,s_][2S+sprimei_possible])
        opt = Opt(:LN_COBYLA, 2S_possible)
        equality_constraint!(opt, cons_no_trans, zeros(S_possible))
    end
    init[init.> ub] = ub[init.> ub]
    init[init.< lb] = lb[init.< lb]

    min_objective!(opt, objf)
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)
    maxeval!(opt, 10000000)
    maxtime!(opt, 10)
    ftol_rel!(opt, 1e-8)
    ftol_abs!(opt, 1e-8)

    (minf,minx,ret) = optimize(opt, init)

    if ret != :SUCCESS && ret != :ROUNDOFF_LIMITED && ret != :MAXEVAL_REACHED &&
         ret != :FTOL_REACHED && ret != :MAXTIME_REACHED
        error("optimization failed: ret = $ret")
    end

    T.z0[i_x,s_][sprimei_possible] = minx[1:S_possible]
    T.z0[i_x,s_][S+sprimei_possible] = minx[1:S_possible]+G[sprimei_possible]
    T.z0[i_x,s_][2S+sprimei_possible] = minx[S_possible+1:2S_possible]
    if para.transfers == true
        T.z0[i_x,s_][3S+sprimei_possible] = minx[2S_possible+1:3S_possible]
    else
        T.z0[i_x,s_][3S+sprimei_possible] = zeros(S)
    end

    return vcat(-minf, T.z0[i_x,s_])
end

"""
Finds the optimal policies
"""
function get_policies_time0(T::BellmanEquation,
        B_::Real,s0::Integer,Vf::Array{Function},xbar::Vector)
    para = T.para
    beta, Theta, G = para.beta, para.Theta, para.G
    U, Uc, Un = para.U, para.Uc, para.Un

    function objf(z,grad)
        c, xprime = z[1], z[2]
        n=(c+G[s0])/Theta[s0]
        return -(U(c,n)+beta*Vf[s0](xprime))
    end

    function cons(z,grad)
        c, xprime, TT = z[1], z[2], z[3]
        n=(c+G[s0])/Theta[s0]
        return -Uc(c,n)*(c-B_-TT)-Un(c,n)*n - beta*xprime
    end
    cons_no_trans(z,grad) = cons(vcat(z,0.0), grad)

    if para.transfers == true
        lb = [0.0, xbar[1], 0.0]
        if para.n_less_than_one == true
            ub = [1-G[s0], xbar[2], 100]
        else
            ub = [100.0, xbar[2], 100.0]
        end
        init = vcat(T.zFB[s0][1], T.zFB[s0][3], T.zFB[s0][4])
        init = [0.95124922, -1.15926816,  0.0]
        opt = Opt(:LN_COBYLA, 3)
        equality_constraint!(opt, cons)
    else
        lb = [0.0, xbar[1]]
        if para.n_less_than_one == true
            ub = [1-G[s0], xbar[2]]
        else
            ub = [100, xbar[2]]
        end
        init = vcat(T.zFB[s0][1], T.zFB[s0][3])
        init = [0.95124922, -1.15926816]
        opt = Opt(:LN_COBYLA, 2)
        equality_constraint!(opt, cons_no_trans)
    end
    init[init.> ub] = ub[init.> ub]
    init[init.< lb] = lb[init.< lb]


    min_objective!(opt, objf)
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)
    maxeval!(opt, 100000000)
    maxtime!(opt, 30)
    # ftol_rel!(opt, 1e-16)
    # ftol_abs!(opt, 1e-16)

    (minf,minx,ret) = optimize(opt, init)

    if ret != :SUCCESS && ret != :ROUNDOFF_LIMITED && ret != :MAXEVAL_REACHED && ret != :FTOL_REACHED
        error("optimization failed: ret = $ret")
    end

    if para.transfers == true
        return -minf, minx[1], minx[1]+G[s0], minx[2], minx[3]
    else
        return -minf, minx[1], minx[1]+G[s0], minx[2], 0.0
    end
end
