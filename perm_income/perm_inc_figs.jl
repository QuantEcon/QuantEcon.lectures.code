#=

@author : Spencer Lyon

=#

using Plots
pyplot()

const r = 0.05
const beta = 1.0 / (1.0 + r)
const T = 60
const sigma = 0.15
const mu = 1.0


function time_path2()
    w = randn(T+1)
    w[1] =  0.0
    b = Array{Float64}(T+1)
    for t=2:T+1
        b[t] = sum(w[1:t])
    end
    b .*= -sigma
    c = mu + (1.0 - beta) .* (sigma .* w .- b)
    return w, b, c
end


# == Figure showing a typical realization == #
function single_realization()
    w, b, c = time_path2()
    p = plot(0:T, mu + sigma .* w, color=:green, label="non-financial income")
    plot!(c, color=:black, label="consumption")
    plot!(b, color=:blue, label="debt")
    plot!(xlabel="Time", linewidth=2, alpha=0.7, xlims=(0, T))

    return p
end


# == Figure showing multiple consumption paths == #
function consumption_paths(n=250)  # n is number of paths
    time_paths = []
    
    for i=1:n
        push!(time_paths, time_path2()[3])
    end

    p = plot(time_paths, linewidth=0.8, alpha=0.7, legend=:none)
    plot!(xlabel="Time", ylabel="Consumption", xlims=(0, T))
    return p
end
