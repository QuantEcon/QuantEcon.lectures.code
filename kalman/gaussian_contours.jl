#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#
using Plots
pyplot()
using LaTeXStrings


# bivariate normal function. I could call plt.mlab, but this is more fun!
# See http://mathworld.wolfram.com/BivariateNormalDistribution.html
function bivariate_normal(X::Matrix, Y::Matrix, σ_x::Real=1.0, σ_y::Real=1.0,
                          μ_x::Real=0.0, μ_y::Real=0.0, σ_xy::Real=0.0)
    Xμ = X .- μ_x
    Yμ = Y .- μ_y

    ρ = σ_xy/(σ_x*σ_y)
    z = Xμ.^2/σ_x^2 + Yμ.^2/σ_y^2 - 2*ρ.*Xμ.*Yμ/(σ_x*σ_y)
    denom = 2π*σ_x*σ_y*sqrt(1-ρ^2)
    return exp.(-z/(2*(1-ρ^2))) ./ denom
end


# == Set up the Gaussian prior density p == #
Σ = [0.4 0.3
     0.3 0.45]
x_hat = [0.2
         -0.2]''

# == Define the matrices G and R from the equation y = G x + N(0, R) == #
G = eye(2)
R = 0.5 .* Σ

# == The matrices A and Q == #
A = [1.2 0
     0   -0.2]
Q = 0.3 .* Σ

# == The observed value of y == #
y = [2.3, -1.9]''

# == Set up grid for plotting == #
x_grid = linspace(-1.5, 2.9, 100)
y_grid = linspace(-3.1, 1.7, 100)
X=repmat(x_grid',length(y_grid),1)
Y=repmat(y_grid,1,length(y_grid))

function gen_gaussian_plot_vals(μ, C)
    "Z values for plotting the bivariate Gaussian N(μ, C)"
    m_x, m_y = μ[1], μ[2]
    s_x, s_y = sqrt(C[1, 1]), sqrt(C[2, 2])
    s_xy = C[1, 2]
    return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)
end


function plot1()
    Z = gen_gaussian_plot_vals(x_hat, Σ)
    contour(x_grid, y_grid, Z, fill=true, levels=6, color=:lightrainbow)
    contour!(x_grid, y_grid, Z, fill=false, levels=6, color=:grays, cbar=false)
end


function plot2()
    Z = gen_gaussian_plot_vals(x_hat, Σ)
    contour(x_grid, y_grid, Z, fill=true, levels=6, color=:lightrainbow)
    contour!(x_grid, y_grid, Z, fill=false, levels=6, color=:grays, cbar=false)
    annotate!(y[1], y[2], L"$y$", color=:black)
end


function plot3()
    Z = gen_gaussian_plot_vals(x_hat, Σ)
    M = Σ * G' * inv(G * Σ * G' + R)
    x_hat_F = x_hat + M * (y - G * x_hat)
    Σ_F = Σ - M * G * Σ
    new_Z = gen_gaussian_plot_vals(x_hat_F, Σ_F)
    # Plot Density 1
    contour(x_grid, y_grid, new_Z, fill=true, levels=6, color=:lightrainbow, alpha=0.8)
    contour!(x_grid, y_grid, new_Z, fill=false, levels=6, color=:grays, cbar=false)
    # Plot Density 2
    contour!(x_grid, y_grid, Z, fill=false, levels=6, color=:grays, cbar=false)
    annotate!(y[1], y[2], L"$y$", color=:black)
end


function plot4()
    Z = gen_gaussian_plot_vals(x_hat, Σ)
    M = Σ * G' * inv(G * Σ * G' + R)
    x_hat_F = x_hat + M * (y - G * x_hat)
    Σ_F = Σ - M * G * Σ
    Z_F = gen_gaussian_plot_vals(x_hat_F, Σ_F)
    new_x_hat = A * x_hat_F
    new_Σ = A * Σ_F * A' + Q
    new_Z = gen_gaussian_plot_vals(new_x_hat, new_Σ)
    # Plot Density 1
    contour(x_grid, y_grid, new_Z, fill=true, levels=6, color=:lightrainbow, alpha=0.9)
    contour!(x_grid, y_grid, new_Z, fill=false, levels=6, color=:grays, cbar=false)
    # Plot Density 2
    contour!(x_grid, y_grid, Z, fill=false, levels=6, color=:grays, cbar=false)
    # Plot Density 3
    contour!(x_grid, y_grid, Z_F, fill=false, levels=6, color=:grays, cbar=false)
    annotate!(y[1], y[2], L"$y$", color=:black)
end
