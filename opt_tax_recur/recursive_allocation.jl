
"""
Compute the planner's allocation by solving Bellman
equation.
"""
struct Planners_Allocation_Bellman{TP <: Para, TI <: Integer,
                                   TVg <: AbstractVector, TVv <: AbstractVector,
                                   TVp <: AbstractArray}
    para::TP
    mc::MarkovChain
    S::TI
    T::BellmanEquation
    mugrid::TVg
    xgrid::TVg
    Vf::TVv
    policies::TVp
end

"""
Initializes the class from the calibration `Para`
"""
function Planners_Allocation_Bellman(para::Para, mugrid::AbstractArray)
    mc = MarkovChain(para.Pi)
    G = para.G
    S = size(para.Pi, 1) # number of states
    #now find the first best allocation
    Vf, policies, T, xgrid = solve_time1_bellman(para, mugrid)
    T.time_0 = true #Bellman equation now solves time 0 problem
    return Planners_Allocation_Bellman(para, mc, S, T, mugrid, xgrid, Vf, policies)
end

"""
Solve the time 1 Bellman equation for calibration `Para` and initial grid `mugrid0`
"""
function solve_time1_bellman{TF <: AbstractFloat}(para::Para{TF}, mugrid::AbstractArray)
    mugrid0 = mugrid
    S = size(para.Pi, 1)
    #First get initial fit
    PP = Planners_Allocation_Sequential(para)
    c = Matrix{TF}(length(mugrid), 2)
    n = Matrix{TF}(length(mugrid), 2)
    x = Matrix{TF}(length(mugrid), 2)
    V = Matrix{TF}(length(mugrid), 2)
    for (i, mu) in enumerate(mugrid0)
        c[i, :], n[i, :], x[i, :], V[i, :] = time1_value(PP, mu)
    end
    Vf = Vector{LinInterp}(2)
    cf = Vector{LinInterp}(2)
    nf = Vector{LinInterp}(2)
    xprimef = Array{LinInterp}(2, S)
    for s in 1:2
        cf[s] = LinInterp(x[:, s][end:-1:1], c[:, s][end:-1:1])
        nf[s] = LinInterp(x[:, s][end:-1:1], n[:, s][end:-1:1])
        Vf[s] = LinInterp(x[:, s][end:-1:1], V[:, s][end:-1:1])
        for sprime in 1:S
            xprimef[s, sprime] = LinInterp(x[:, s][end:-1:1], x[:, s][end:-1:1])
        end
    end
    policies = [cf, nf, xprimef]
    #create xgrid
    xbar = [maximum(minimum(x, 1)), minimum(maximum(x, 1))]
    xgrid = linspace(xbar[1], xbar[2], length(mugrid0))
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
            diff = max(diff, maximum(abs, (Vf[s].(xgrid)-Vfnew[s].(xgrid))./Vf[s].(xgrid)))
        end
        print("diff = $diff \n")
        Vf = Vfnew
    end
    # store value function policies and Bellman Equations
    return Vf, policies, T, xgrid
end

"""
Fits the policy functions PF using the points `xgrid` using interpolation
"""
function fit_policy_function(PP::Planners_Allocation_Sequential,
                            PF::Function, xgrid::AbstractArray)
    S = PP.S
    Vf = Vector{LinInterp}(S)
    cf = Vector{LinInterp}(S)
    nf = Vector{LinInterp}(S)
    xprimef = Array{LinInterp}(S, S)
    for s in 1:S
        PFvec = Array{typeof(PP.para).parameters[1]}(length(xgrid), 3+S)
        for (i_x, x) in enumerate(xgrid)
            PFvec[i_x, :] = PF(i_x, x, s)
        end
        Vf[s] = LinInterp(xgrid, PFvec[:, 1])
        cf[s] = LinInterp(xgrid, PFvec[:, 2])
        nf[s] = LinInterp(xgrid, PFvec[:, 3])
        for sprime in 1:S
            xprimef[s, sprime] = LinInterp(xgrid, PFvec[:, 3+sprime])
        end
    end
    return Vf, [cf, nf, xprimef]
end

"""
Finds the optimal allocation given initial government debt `B_` and state `s_0`
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
    c0, n0, xprime0 = z0[2], z0[3], z0[4:end]
    return c0, n0, xprime0
end

"""
Simulates Ramsey plan for `T` periods
"""
function simulate(pab::Planners_Allocation_Bellman,
                B_::AbstractFloat, s_0::Integer, T::Integer,
                sHist::Vector=QuantEcon.simulate(mc, s_0, T))
    para, S, policies = pab.para, pab.S, pab.policies
    beta, Pi, Uc = para.beta, para.Pi, para.Uc
    cf, nf, xprimef = policies[1], policies[2], policies[3]
    TF = typeof(para).parameters[1]
    cHist = Vector{TF}(T)
    nHist = Vector{TF}(T)
    Bhist = Vector{TF}(T)
    TauHist = Vector{TF}(T)
    muHist = Vector{TF}(T)
    RHist = Vector{TF}(T-1)
    #time0
    cHist[1], nHist[1], xprime = time0_allocation(pab, B_, s_0)
    TauHist[1] = Tau(pab.para, cHist[1], nHist[1])[s_0]
    Bhist[1] = B_
    muHist[1] = 0.0
    #time 1 onward
    for t in 2:T
        s, x = sHist[t], xprime[sHist[t]]
        n = nf[s](x)
        c = [cf[shat](x) for shat in 1:S]
        xprime = [xprimef[s, sprime](x) for sprime in 1:S]
        TauHist[t] = Tau(pab.para, c, n)[s]
        u_c = Uc(c, n)
        Eu_c = dot(Pi[sHist[t-1], :], u_c)
        muHist[t] = pab.Vf[s](x)
        RHist[t-1] = Uc(cHist[t-1], nHist[t-1])/(beta*Eu_c)
        cHist[t], nHist[t], Bhist[t] = c[s], n, x/u_c[s]
    end
    return cHist, nHist, Bhist, TauHist, sHist, muHist, RHist
end
