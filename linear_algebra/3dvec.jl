
using Plots
#plotlyjs()
pyplot()

x_min, x_max = -5, 5
y_min, y_max = -5, 5

alpha, beta = 0.2, 0.1

# Axes
gs = 3
z = linspace(x_min, x_max, gs)
x = zeros(gs)
y = zeros(gs)
plot(x, y, z, color=:black, linewidth=2, alpha=0.5, label="")
plot!(z, x, y, color=:black, linewidth=2, alpha=0.5, label="")
plot!(y, z, x, color=:black, linewidth=2, alpha=0.5, label="")

# Fixed linear function, to generate a plane
f(x, y) = alpha .* x + beta .* y

# Vector locations, by coordinate
x_coords = [3, 3]
y_coords = [4, -4]
z = f(x_coords, y_coords)

# Lines to vectors
n = 2
x_vec = zeros(n, n)
y_vec = zeros(n, n)
z_vec = zeros(n, n)
labels = []
for i=1:n
  x_vec[:, i] = [0; x_coords[i]]
  y_vec[:, i] = [0; y_coords[i]]
  z_vec[:, i] = [0; f(x_coords[i], y_coords[i])]
  lab = string("a", i)
  push!(labels, lab)
end
plot!(x_vec, y_vec, z_vec, color=[:blue :red], linewidth=1.5, alpha=0.6, label=reshape(labels,1,length(labels)))

# Draw the plane
grid_size = 20
xr2 = linspace(x_min, x_max, grid_size)
yr2 = linspace(y_min, y_max, grid_size)
z2 = Array{Float64}(grid_size, grid_size)
for i in 1:grid_size
    for j in 1:grid_size
        z2[j, i] = f(xr2[i], yr2[j])
    end
end
surface!(xr2, yr2, z2, cbar=false, alpha=0.2, fill=:blues, xlims=(x_min, x_max), ylims=(x_min, x_max), zlims=(x_min, x_max), xticks=[0], yticks=[0], zticks=[0])
