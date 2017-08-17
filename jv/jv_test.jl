using QuantEcon
using LaTeXStrings
using Plots
pyplot()


wp = JvWorker(grid_size=25)
v_init = wp.x_grid .* 0.5

f(x) = bellman_operator(wp, x)
V = compute_fixed_point(f, v_init, max_iter=300)

s_policy, phi_policy = bellman_operator(wp, V, ret_policies=true)

# === plot solution === #
p=plot(wp.x_grid, [phi_policy s_policy V],
  title = ["phi policy" "s policy" "value function"],
  color = [:orange :blue :green], width = 3,
  xaxis = ("x", (0.0, maximum(wp.x_grid))),
  yaxis = ((-0.1, 1.1)),
  size = (800, 800), 
  legend = false,
  layout = (3,1), bottom_margin = 20mm, show = false)
