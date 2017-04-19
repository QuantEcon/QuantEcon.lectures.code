using QuantEcon
using Plots
pyplot()

# == Plot functions == #

function plot_spectral_density(arma::ARMA)
    (w, spect) = spectral_density(arma, two_pi=false)
    p = plot(w, spect, color=:blue, linewidth=2, alpha=0.7,
             xlims=(0, pi), xlabel="frequency", ylabel="spectrum",
             title="Spectral density", yscale=:log, legend=:none, grid=false)
    return p
end


function plot_autocovariance(arma::ARMA)
    acov = autocovariance(arma)
    n = length(acov)
    N = repmat(0:(n - 1), 1, 2)'
    heights = [zeros(1,n); acov']
    p = scatter(0:(n - 1), acov, title="Autocovariance", xlims=(-0.5, n - 0.5),
                xlabel="time", ylabel="autocovariance", legend=:none, color=:blue)
    plot!(-1:(n + 1), zeros(1, n + 3), color=:red, linewidth=0.5)
    plot!(N, heights, color=:blue, grid=false)
    return p
end

function plot_impulse_response(arma::ARMA)
    psi = impulse_response(arma)
    n = length(psi)
    N = repmat(0:(n - 1), 1, 2)'
    heights = [zeros(1,n); psi']
    p = scatter(0:(n - 1), psi, title="Impulse response", xlims=(-0.5, n - 0.5),
                xlabel="time", ylabel="response", legend=:none, color=:blue)
    plot!(-1:(n + 1), zeros(1, n + 3), color=:red, linewidth=0.5)
    plot!(N, heights, color=:blue, grid=false)
    return p
end

function plot_simulation(arma::ARMA)
    X = simulation(arma)
    n = length(X)
    p = plot(0:(n - 1), X, color=:blue, linewidth=2, alpha=0.7,
             xlims=(0.0, n), xlabel="time", ylabel="state space",
            title="Sample path", legend=:none, grid=:false)
    return p
end

function quad_plot(arma::ARMA)
    p = plot(plot_impulse_response(arma), plot_autocovariance(arma),
         plot_spectral_density(arma), plot_simulation(arma))
    return p
end
