#=

main_LS.jl

@author: Shunsuke Hori

=#
using LS
using CES
using BGP

"""
Time Varying Example
"""
PP_seq_time = LS.Planners_Allocation_Sequential(CES.M_time_example) #solve sequential problem

sHist_h = [1, 2, 3, 4, 6, 6, 6]
sHist_l = [1, 2, 3, 5, 6, 6, 6]

sim_seq_h = LS.simulate(PP_seq_time, 1.0, 1, 7, sHist_h)
sim_seq_l = LS.simulate(PP_seq_time, 1.0, 1, 7, sHist_l)

using PyPlot

plt[:figure](figsize=[14, 10])
plt[:subplot](3, 2, 1)
plt[:title]("Consumption")
plt[:plot](sim_seq_l[1], "-ok")
plt[:plot](sim_seq_h[1], "-or")
plt[:subplot](3, 2, 2)
plt[:title]("Labor Supply")
plt[:plot](sim_seq_l[2], "-ok")
plt[:plot](sim_seq_h[2], "-or")
plt[:subplot](3, 2, 3)
plt[:title]("Government Debt")
plt[:plot](sim_seq_l[3], "-ok")
plt[:plot](sim_seq_h[3], "-or")
plt[:subplot](3, 2, 4)
plt[:title]("Tax Rate")
plt[:plot](sim_seq_l[4], "-ok")
plt[:plot](sim_seq_h[4], "-or")
plt[:subplot](3, 2, 5)
plt[:title]("Government Spending")
plt[:plot](CES.M_time_example.G[sHist_l], "-ok")
plt[:plot](CES.M_time_example.G[sHist_h], "-or")
plt[:subplot](3, 2, 6)
plt[:title]("Output")
plt[:plot](CES.M_time_example.Theta[sHist_l].*sim_seq_l[2], "-ok")
plt[:plot](CES.M_time_example.Theta[sHist_h].*sim_seq_h[2], "-or")

plt[:tight_layout]()
plt[:savefig]("TaxSequence_time_varying.png")

plt[:figure](figsize=[8, 5])
plt[:title]("Gross Interest Rate")
plt[:plot](sim_seq_l[end], "-ok")
plt[:plot](sim_seq_h[end], "-or")
plt[:tight_layout]()
plt[:savefig]("InterestRate_time_varying.png")


"""
Time 0 example
"""
PP_seq_time0 = LS.Planners_Allocation_Sequential(CES.M2) #solve sequential problem

B_vec = linspace(-1.5, 1.0, 100)
taxpolicy = hcat([LS.simulate(PP_seq_time0, B_, 1, 2)[4] for B_ in B_vec]...)'
interest_rate = hcat([LS.simulate(PP_seq_time0, B_, 1, 3)[end] for B_ in B_vec]...)'

plt[:figure](figsize=(14, 6))
plt[:subplot](211)
plt[:plot](B_vec,taxpolicy[:, 1], linewidth=2.)
plt[:plot](B_vec,taxpolicy[:, 2], linewidth=2.)

plt[:title]("Tax Rate")
plt[:legend]((latexstring("Time ", "t=0"),
        latexstring("Time ", L"t \geq 1")), loc=2, shadow=true)
plt[:subplot](212)
plt[:title]("Gross Interest Rate")
plt[:plot](B_vec,interest_rate[:, 1], linewidth=2.)
plt[:plot](B_vec,interest_rate[:, 2], linewidth=2.)
plt[:xlabel]("Initial Government Debt")
plt[:tight_layout]()

plt[:savefig]("Time0_taxpolicy.png")

#compute the debt entered with at time 1
B1_vec = hcat([LS.simulate(PP_seq_time0, B_, 1, 2)[3][2] for B_ in B_vec]...)'
#now compute the optimal policy if the government could reset
tau1_reset = hcat([LS.simulate(PP_seq_time0, B1, 1, 1)[4] for B1 in B1_vec]...)'

plt[:figure](figsize=[10, 6])
plt[:plot](B_vec, taxpolicy[:, 2], linewidth=2.)
plt[:plot](B_vec, tau1_reset, linewidth=2.)
plt[:xlabel]("Initial Government Debt")
plt[:title]("Tax Rate")
plt[:legend]((L"\tau_1", L"\tau_1^R"), loc=2, shadow=true)
plt[:tight_layout]()

plt[:savefig]("Time0_inconsistent.png")


"""
BGP Example
"""
#initialize mugrid for value function iteration
muvec = linspace(-0.6, 0.0, 200)


PP_seq = LS.Planners_Allocation_Sequential(BGP.M1) #solve sequential problem

PP_bel = LS.Planners_Allocation_Bellman(BGP.M1,muvec) #solve recursive problem

T = 20
#sHist = utilities.simulate_markov(M1.Pi,0,T)
sHist = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]

#simulate
sim_seq = LS.simulate(PP_seq, 0.5, 1, T, sHist)
sim_bel = LS.simulate(PP_bel, 0.5, 1, T, sHist)

#plot policies
plt[:figure](figsize=[14, 10])
plt[:subplot](3, 2, 1)
plt[:title]("Consumption")
plt[:plot](sim_seq[1], "-ok")
plt[:plot](sim_bel[1], "-xk")
plt[:legend](("Sequential", "Recursive"), loc="best")
plt[:subplot](3, 2, 2)
plt[:title]("Labor Supply")
plt[:plot](sim_seq[2], "-ok")
plt[:plot](sim_bel[2], "-xk")
plt[:subplot](3, 2, 3)
plt[:title]("Government Debt")
plt[:plot](sim_seq[3], "-ok")
plt[:plot](sim_bel[3], "-xk")
plt[:subplot](3, 2, 4)
plt[:title]("Tax Rate")
plt[:plot](sim_seq[4], "-ok")
plt[:plot](sim_bel[4], "-xk")
plt[:subplot](3, 2, 5)
plt[:title]("Government Spending")
plt[:plot](BGP.M1.G[sHist], "-ok")
plt[:plot](BGP.M1.G[sHist], "-xk")
plt[:plot](BGP.M1.G[sHist], "-^k")
plt[:subplot](3, 2, 6)
plt[:title]("Output")
plt[:plot](BGP.M1.Theta[sHist].*sim_seq[2], "-ok")
plt[:plot](BGP.M1.Theta[sHist].*sim_bel[2], "-xk")

plt[:tight_layout]()
plt[:savefig]("TaxSequence_LS.png")

plt[:figure](figsize=[8, 5])
plt[:title]("Gross Interest Rate")
plt[:plot](sim_seq[end], "-ok")
plt[:plot](sim_bel[end], "-xk")
plt[:legend](("Sequential", "Recursive"), loc="best")
plt[:tight_layout]()
