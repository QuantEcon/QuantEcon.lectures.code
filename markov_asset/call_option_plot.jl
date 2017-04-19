#=
Compare call option prices to consol price.

=#

using Plots
include("asset_pricing.jl")
plotly()

ap = AssetPriceModel(beta=0.9)
zeta = 1.0
strike_price = 40.0

x = ap.mc.state_values
p = consol_price(ap, zeta)
w = call_option(ap, zeta, strike_price)

plot(x, p, color="blue", lw=2, xlabel="state", label="consol price")
plot!(x, w, color="green", lw=2, label="value of call option")
