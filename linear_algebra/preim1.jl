#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#
using Plots
pyplot()
using LaTeXStrings

f(x) = 0.6 * cos(4.0 * x) + 1.3

xmin, xmax = -1.0, 1.0
Nx = 160
x = linspace(xmin, xmax, Nx)
y = f.(x)
ya, yb = minimum(y), maximum(y)

p1 = plot(x, y, color=:black, label=[L"$f$" ""], grid=false)
plot!(x, ya*ones(Nx, 1), fill_between=yb*ones(Nx, 1),
     fillalpha=0.1, color=:blue, label="", lw=0)
plot!(zeros(2, 2), [ya ya; yb yb], lw=3, color=:blue, label=[L"range of $f$" ""])
annotate!(0.04, -0.3, L"$0$", ylims=(-0.6, 3.2))
vline!([0], color=:black, label="")
hline!([0], color=:black, label="")
plot!(foreground_color_axis=:white, foreground_color_text=:white,
      foreground_color_border=:white)

ybar = 1.5
plot!(x, x .* 0 .+ ybar, color=:black, linestyle=:dash, label="")
annotate!(0.05, 0.8 * ybar, L"$y$")

x_vals = Array{Float64}(2, 4)
y_vals = Array{Float64}(2, 4)
labels = []
for (i, z) in enumerate([-0.35, 0.35])
  x_vals[:, 2*i-1] = z*ones(2, 1)
  y_vals[2, 2*i-1] = f(z)
  labels = [labels; (z, -0.2, LaTeXString("\$x_$i\$"))]
end
plot!(x_vals, y_vals, color=:black, linestyle=:dash, label="", annotation=labels)

p2 = plot(x, y, color=:black, label=[L"$f$" ""], grid=false)
plot!(x, ya*ones(Nx, 1), fill_between=yb*ones(Nx, 1),
     fillalpha=0.1, color=:blue, label="", lw=0)
plot!(zeros(2, 2), [ya ya; yb yb], lw=3, color=:blue, label=[L"range of $f$" ""])
annotate!(0.04, -0.3, L"$0$", ylims=(-0.6, 3.2))
vline!([0], color=:black, label="")
hline!([0], color=:black, label="")
plot!(foreground_color_axis=:white, foreground_color_text=:white,
      foreground_color_border=:white)

ybar = 2.6
plot!(x, x .* 0 .+ ybar, color=:black, linestyle=:dash, legend=:none)
annotate!(0.04, 0.91 * ybar, L"$y$")

plot(p1, p2, layout=(2, 1), size=(600,700))


