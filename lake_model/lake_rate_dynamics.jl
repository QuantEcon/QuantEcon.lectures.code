lm = LakeModel()
e_0 = 0.92     # Initial employment rate
u_0 = 1 - e_0  # Initial unemployment rate
T = 50         # Simulation length

xbar = rate_steady_state(lm)
x_0 = [e_0; u_0]
x_path = simulate_rate_path(lm, x_0, T)

titles = ["Employment rate" "Unemployment rate"]
dates = collect(1:T)

plot(dates, x_path', layout=(2, 1), title=titles, legend=:none)
hline!(xbar', layout=(2, 1), color=:red, linestyle=:dash,
       ylims=[0.9999*minimum(x_path[1,:]) Inf;
       -Inf 1.001*maximum(x_path[2,:])])
    
