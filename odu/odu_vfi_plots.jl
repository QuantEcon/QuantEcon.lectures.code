#=

Author: Victoria Gregory

=#
using LaTeXStrings

# Set up the problem and initial guess, solve by VFI
sp = SearchProblem(;w_grid_size=100, pi_grid_size=100)
v_init = zeros(sp.n_w, sp.n_pi) + sp.c / (1 - sp.bet)
f(x) = bellman_operator(sp, x)
v = compute_fixed_point(f, v_init)
policy = get_greedy(sp, v)

# Make functions for the linear interpolants of these
vf = extrapolate(interpolate((sp.w_grid, sp.pi_grid), v, Gridded(Linear())),
                 Flat())
pf = extrapolate(interpolate((sp.w_grid, sp.pi_grid), policy,
                 Gridded(Linear())), Flat())

function plot_value_function(;w_plot_grid_size::Integer=100,
                             pi_plot_grid_size::Integer=100)
    pi_plot_grid = linspace(0.001, 0.99, pi_plot_grid_size)
    w_plot_grid = linspace(0, sp.w_max, w_plot_grid_size)
    Z = [vf[w_plot_grid[j], pi_plot_grid[i]]
            for j in 1:w_plot_grid_size, i in 1:pi_plot_grid_size]
    p = contour(pi_plot_grid, w_plot_grid, Z, levels=12, alpha=0.6, fill=true)
    plot!(xlabel=L"$\pi$", ylabel="wage", xguidefont=font(12))
    return p
end

function plot_policy_function(;w_plot_grid_size::Integer=100,
                             pi_plot_grid_size::Integer=100)
    pi_plot_grid = linspace(0.001, 0.99, pi_plot_grid_size)
    w_plot_grid = linspace(0, sp.w_max, w_plot_grid_size)
    Z = [pf[w_plot_grid[j], pi_plot_grid[i]]
            for j in 1:w_plot_grid_size, i in 1:pi_plot_grid_size]
    p = contour(pi_plot_grid, w_plot_grid, Z, levels=1, alpha=0.6, fill=true)
    plot!(xlabel=L"$\pi$", ylabel="wage", xguidefont=font(12), cbar=false)
    annotate!(0.4, 1.0, "reject")
    annotate!(0.7, 1.8, "accept")
    return p
end
