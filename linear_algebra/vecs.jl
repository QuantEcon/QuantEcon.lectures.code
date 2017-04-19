
#=
Illustrates vectors in the plane.

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

@date: 07/09/2014
=#

using Plots
pyplot()
using LaTeXStrings


function plane_fig()

  vecs = ([2, 4], [-3, 3], [-4, -3.5])
  x_vals = zeros(2, length(vecs))
  y_vals = zeros(2, length(vecs))
  labels = []

  # Create matrices of x and y values, labels for plotting
  for i = 1:length(vecs)
    v = vecs[i]
    x_vals[2, i] = v[1]
    y_vals[2, i] = v[2]
    labels = [labels; (1.1 * v[1], 1.1 * v[2], "$v")]
  end

  plot(x_vals, y_vals, arrow=true, color=:blue,
       legend=:none, xlims=(-5, 5), ylims=(-5, 5),
       annotations = labels, xticks=-5:1:5, yticks=-5:1:5)
  vline!([0], color=:black)
  hline!([0], color=:black)
  plot!(foreground_color_axis=:white, foreground_color_text=:white,
        foreground_color_border=:white)

end


function scalar_multiply()  # illustrate scalar multiplication

  x = [2, 2]
  scalars = [-2, 2]

  # Create matrices of x and y values, labels for plotting
  x_vals = zeros(2, 1 + length(scalars))
  y_vals = zeros(2, 1 + length(scalars))
  labels = []
  x_vals[2, 3] = x[1]
  y_vals[2, 3] = x[2]
  labels = [labels; (x[1] + 0.4, x[2] - 0.2, L"$x$")]

  # Perform scalar multiplication, store results in plotting matrices
  for i = 1:length(scalars)
    s = scalars[i]
    v = s .* x
    x_vals[2, i] = v[1]
    y_vals[2, i] = v[2]
    labels = [labels; (v[1] + 0.4, v[2] - 0.2, LaTeXString("\$$s x\$"))]
  end

  plot(x_vals, y_vals, arrow=true, color=[:red :red :blue],
       legend=:none, xlims=(-5, 5), ylims=(-5, 5),
       annotations = labels, xticks=-5:1:5, yticks=-5:1:5)
  vline!([0], color=:black)
  hline!([0], color=:black)
  plot!(foreground_color_axis=:white, foreground_color_text=:white,
        foreground_color_border=:white)

end

# Plot the first figure --- three vectors in the plane

plane_fig()


