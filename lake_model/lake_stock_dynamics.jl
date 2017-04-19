#=
Stock dynamics the a lake model.

=#

using Plots
pyplot()
include("lake_model.jl")

lm = LakeModel()
N_0 = 150      # Population
e_0 = 0.92     # Initial employment rate
u_0 = 1 - e_0  # Initial unemployment rate
T = 50         # Simulation length

E_0 = e_0 * N_0
U_0 = u_0 * N_0
X_0 = [E_0; U_0]

X_path = simulate_stock_path(lm, X_0, T)

titles = ["Employment" "Unemployment" "Labor Force"]
dates = collect(1:T)

x1 = squeeze(X_path[1, :], 1)
x2 = squeeze(X_path[2, :], 1)
x3 = squeeze(sum(X_path, 1), 1)

plot(dates, Vector[x1, x2, x3], title=titles, layout=(3, 1), legend=:none)
