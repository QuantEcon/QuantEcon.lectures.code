
using Plots
pyplot()
using QuantEcon 

P =[0.971 0.029 0.000
    0.145 0.778 0.077
    0.000 0.508 0.492]

psi = [0.0 0.2 0.8]

t = 20
x_vals = Array{Float64}(t+1)
y_vals = Array{Float64}(t+1)
z_vals = Array{Float64}(t+1)
colors = []

for i=1:t
    x_vals[i] = psi[1]
    y_vals[i] = psi[2]
    z_vals[i] = psi[3]
    psi = psi*P
    push!(colors, :red)
end
push!(colors, :black)

mc = MarkovChain(P)
psi_star = stationary_distributions(mc)[1]
x_vals[t+1] = psi_star[1]
y_vals[t+1] = psi_star[2]
z_vals[t+1] = psi_star[3]
scatter(x_vals, y_vals, z_vals, color=colors)
plot!(lims=(0, 1), ticks=[0.25 0.5 0.75]', legend=:none)
