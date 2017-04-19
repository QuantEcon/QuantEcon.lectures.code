
#=
Plots of Lucas tree price for different gamma.

=#

using Plots, LaTeXStrings
pyplot()


gammas = [1.2, 1.4, 1.6, 1.8, 2.0]
ap = AssetPriceModel()
states = ap.mc.state_values

lines = []
labels = []

for gamma in gammas
    ap.gamma = gamma
    v = tree_price(ap)
    label="gamma = $gamma"
    push!(labels, label)
    push!(lines, v)
end

plot(lines, 
    labels=reshape(labels,1,length(labels)), 
    title="Price-divdend ratio as a function of the state",
    ylabel="price-dividend ratio",
    xlabel="state")


