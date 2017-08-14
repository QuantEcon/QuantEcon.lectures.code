grid_size = 25  
beta_vals = linspace(0.8, 0.99, grid_size)  
w_bar_vals = similar(beta_vals)

mcm = McCallModel()

for (i, beta) in enumerate(beta_vals)
    mcm.beta = beta
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar
end

plot(beta_vals, 
    w_bar_vals, 
    lw=2, 
    alpha=0.7, 
    xlabel="discount rate",
    ylabel="reservation wage",
    label=L"$\bar w$ as a function of $\beta$")

