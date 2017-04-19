
ode:: julia

#=

Plots reservation wage against the job separation rate

=#


grid_size = 25  
alpha_vals = linspace(0.05, 0.5, grid_size)  
w_bar_vals = similar(alpha_vals)

mcm = McCallModel()

for (i, alpha) in enumerate(alpha_vals)
    mcm.alpha = alpha
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar
end

plot(alpha_vals, 
    w_bar_vals, 
    lw=2, 
    alpha=0.7, 
    xlabel="job separation rate",
    ylabel="reservation wage",
    label=L"$\bar w$ as a function of $\alpha$")

