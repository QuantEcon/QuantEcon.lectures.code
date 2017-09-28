#=

Author: Shunsuke Hori

=#

using LS
using CES
using BGP

#initialize mugrid for value function iteration
muvec = linspace(-0.7, 0.01, 200)

#Time Varying Example

CES.M_time_example.transfers = true #Government can use transfers
PP_seq_time = LS.Planners_Allocation_Sequential(CES.M_time_example) #solve sequential problem

PP_im_time = Planners_Allocation_Bellman(CES.M_time_example, muvec)

sHist_h = [1, 2, 3, 4, 6, 6, 6]
sHist_l = [1, 2, 3, 5, 6, 6, 6]

sim_seq_h = LS.simulate(PP_seq_time, 1., 1, 7, sHist_h)
sim_im_h = simulate(PP_im_time, 1., 1, 7, sHist_h)
sim_seq_l = LS.simulate(PP_seq_time, 1., 1, 7, sHist_l)
sim_im_l = simulate(PP_im_time, 1., 1, 7, sHist_l)

using Plots
pyplot()
p=plot(size = (700, 500), layout =(3, 2),
        xaxis=(0:6), grid=false, titlefont=Plots.font("sans-serif", 10))
title!(p[1], "Consumption")
plot!(p[1], 0:6, sim_seq_l[1], marker=:circle, color=:black, legend=false)
plot!(p[1], 0:6, sim_im_l[1], marker=:circle, color=:red, legend=false)
plot!(p[1], 0:6, sim_seq_h[1], marker=:utriangle, color=:black, legend=false)
plot!(p[1], 0:6, sim_im_h[1], marker=:utriangle, color=:red, legend=false)
title!(p[2],"Labor")
plot!(p[2], 0:6, sim_seq_l[2], marker=:circle, color=:black, lab="Complete Markets")
plot!(p[2], 0:6, sim_im_l[2], marker=:circle, color=:red, lab="Incomplete Markets")
plot!(p[2], 0:6, sim_seq_h[2], marker=:utriangle, color=:black, lab="")
plot!(p[2], 0:6, sim_im_h[2], marker=:utriangle, color=:red, lab="")
title!(p[3],"Government Debt")
plot!(p[3], 0:6, sim_seq_l[3], marker=:circle, color=:black, legend=false)
plot!(p[3], 0:6, sim_im_l[3], marker=:circle, color=:red, legend=false)
plot!(p[3], 0:6, sim_seq_h[3], marker=:utriangle, color=:black, legend=false)
plot!(p[3], 0:6, sim_im_h[3], marker=:utriangle, color=:red, legend=false)
title!(p[4],"Tax Rate")
plot!(p[4], 0:6, sim_seq_l[4], marker=:circle, color=:black, legend=false)
plot!(p[4], 0:6, sim_im_l[5], marker=:circle, color=:red, legend=false)
plot!(p[4], 0:6, sim_seq_h[4], marker=:utriangle, color=:black, legend=false)
plot!(p[4], 0:6, sim_im_h[5], marker=:utriangle, color=:red, legend=false)
title!(p[5],"Government Spending", ylims=(0.05,0.25))
plot!(p[5], 0:6, CES.M_time_example.G[sHist_l], marker=:circle, color=:black, legend=false)
plot!(p[5], 0:6, CES.M_time_example.G[sHist_l], marker=:circle, color=:red, legend=false)
plot!(p[5], 0:6, CES.M_time_example.G[sHist_h], marker=:utriangle, color=:black, legend=false)
plot!(p[5], 0:6, CES.M_time_example.G[sHist_h], marker=:utriangle, color=:red, legend=false)
title!(p[6],"Output")
plot!(p[6], 0:6, CES.M_time_example.Theta[sHist_l].*sim_seq_l[2],
        marker=:circle, color=:black, legend=false)
plot!(p[6], 0:6, CES.M_time_example.Theta[sHist_l].*sim_im_l[2],
        marker=:circle, color=:red, legend=false)
plot!(p[6], 0:6, CES.M_time_example.Theta[sHist_h].*sim_seq_h[2],
        marker=:utriangle, color=:black, legend=false)
plot!(p[6], 0:6, CES.M_time_example.Theta[sHist_h].*sim_im_h[2],
        marker=:utriangle, color=:red, legend=false)
savefig("TaxSequence_time_varying_AMSS.png")
p1=p

#=
BGP Example
=#
BGP.M1.transfers = false   #Government can use transfers
PP_seq = LS.Planners_Allocation_Sequential(BGP.M1) #solve sequential problem
PP_bel = LS.Planners_Allocation_Bellman(BGP.M1, muvec) #solve recursive problem
PP_im = Planners_Allocation_Bellman(BGP.M1, muvec)

T = 20
#sHist = utilities.simulate_markov(M1.Pi,0,T)
sHist = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]

#simulate
sim_seq = LS.simulate(PP_seq, 0.5, 1, T, sHist)
#sim_bel = PP_bel.simulate(0.5,0,T,sHist)
sim_im = simulate(PP_im, 0.5, 1, T, sHist)

#plot policies
p=plot(size = (700, 500), layout = grid(3, 2),
        xaxis=(0:T), grid=false, titlefont=Plots.font("sans-serif", 10))
title!(p[1], "Consumption")
plot!(p[1], sim_seq[1], marker=:circle, color=:black, lab="Complete Markets")
#plt[:plot](sim_bel[0],"-xk")
plot!(p[1], sim_im[1], marker=:utriangle, color=:black, lab="Incomplete Markets")
title!(p[2], "Labor")
plot!(p[2], sim_seq[1], marker=:circle, color=:black, legend=false)
#plt.plot(sim_bel[1],"-xk")
plot!(p[2], sim_im[1], marker=:utriangle, color=:black, legend=false)
title!(p[3], "Government Debt")
plot!(p[3], sim_seq[2], marker=:circle, color=:black, legend=false)
#plt.plot(sim_bel[2],"-xk")
plot!(p[3], sim_im[2], marker=:utriangle, color=:black, legend=false)
title!(p[4], "Tax Rate")
plot!(p[4], sim_seq[4], marker=:circle, color=:black, legend=false)
#plt.plot(sim_bel[3],"-xk")
plot!(p[4], sim_im[5], marker=:utriangle, color=:black, legend=false)
title!(p[5], "Government Spending", ylims=(0.05,0.25))
plot!(p[5], BGP.M1.G[sHist], marker=:circle, color=:black, legend=false)
#plt.plot(M1.G[sHist],"-^k")
title!(p[6], "Output")
plot!(p[6], BGP.M1.Theta[sHist].*sim_seq[2], marker=:circle,
        color=:black, legend=false)
#plt.plot(M1.Theta[sHist]*sim_bel[1],"-xk")
plot!(p[6], BGP.M1.Theta[sHist].*sim_im[2], marker=:utriangle,
        color=:black, legend=false)
savefig("TaxSequence_AMSS.png")
p2=p

#Now long simulations
T_long = 200
sim_seq_long = LS.simulate(PP_seq, 0.5, 1, T_long)
sHist_long = sim_seq_long[end-2]
sim_im_long = simulate(PP_im, 0.5, 1, T_long, sHist_long)

p=plot(size = (700, 500), layout = (3, 2), xaxis=(0:50:T_long), grid=false,
        titlefont=Plots.font("sans-serif", 10))
title!(p[1], "Consumption")
plot!(p[1], sim_seq_long[1], color=:black, linestyle=:solid, lab="Complete Markets")
plot!(p[1], sim_im_long[1], color=:black, linestyle=:dot, lab="Incomplete Markets")
title!(p[2], "Labor")
plot!(p[2], sim_seq_long[1], color=:black, linestyle=:solid, legend=false)
plot!(p[2], sim_im_long[1], color=:black, linestyle=:dot, legend=false)
title!(p[3], "Government Debt")
plot!(p[3], sim_seq_long[2], color=:black, linestyle=:solid, legend=false)
plot!(p[3], sim_im_long[2], color=:black, linestyle=:dot, legend=false)
title!(p[4], "Tax Rate")
plot!(p[4], sim_seq_long[3], color=:black, linestyle=:solid, legend=false)
plot!(p[4], sim_im_long[4], color=:black, linestyle=:dot, legend=false)
title!(p[5], "Government Spending",ylims=(0.05,0.25))
plot!(p[5], BGP.M1.G[sHist_long], color=:black, linestyle=:solid, legend=false)
plot!(p[5], BGP.M1.G[sHist_long], color=:black, linestyle=:dot, legend=false)
title!(p[6], "Output")
plot!(p[6], BGP.M1.Theta[sHist_long].*sim_seq_long[2],
    color=:black, linestyle=:solid, legend=false)
plot!(p[6], BGP.M1.Theta[sHist_long].*sim_im_long[2],
    color=:black, linestyle=:dot, legend=false)
# plt[:tight_layout]()
savefig("Long_SimulationAMSS.png")
p3=p

#=
Show Convergence example
=#
M_convergence = CES.M1
muvec = linspace(-0.15, 0.0, 100) #change
PP_C = Planners_Allocation_Bellman(M_convergence, muvec)
xgrid = PP_C.xgrid
xf = PP_C.policies[end-1] #get x policies
p= plot()
for s in 1:2
    plot!(p, xgrid, xf[1, s].(xgrid)-xgrid)
end
p4=p

sim_seq_convergence = simulate(PP_C, 0.5, 1, 2000)
sHist_long = sim_seq_convergence[end]

p=plot(size = (700, 500),layout = grid(3, 2), xaxis=(0:200:2000),
        grid=false, titlefont=Plots.font("sans-serif", 10))
title!("Consumption")
plot!(sim_seq_convergence[1], color=:black,
    lab=["Complete Markets" "Incomplete Markets"])
title!(p[2], "Labor")
plot!(p[2], sim_seq_convergence[2], color=:black, legend=false)
title!(p[3], "Government Debt")
plot!(p[3], sim_seq_convergence[3], color=:black, legend=false)
title!(p[4], "Tax Rate")
plot!(p[4], sim_seq_convergence[4], color=:black, legend=false)
title!(p[5], "Government Spending", ylims=(0.05,0.25))
plot!(p[5], M_convergence.G[sHist_long], color=:black, legend=false)
title!(p[6], "Output")
plot!(p[6], M_convergence.Theta[sHist_long].*sim_seq_convergence[2],
        color=:black, legend=false)
savefig("Convergence_SimulationAMSS.png")
p5=p
