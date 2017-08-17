grid_size = 25  
gamma_vals = linspace(0.05, 0.95, grid_size)  
w_bar_vals = similar(gamma_vals)

mcm = McCallModel()

for (i, gamma) in enumerate(gamma_vals)
    mcm.gamma = gamma
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar
end

plot(gamma_vals, 
    w_bar_vals, 
    lw=2, 
    alpha=0.7, 
    xlabel="job offer rate",
    ylabel="reservation wage",
    label=L"$\bar w$ as a function of $\gamma$")
