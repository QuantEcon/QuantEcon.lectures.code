using PyPlot
# == Set seed and generate a_t sequence == #
srand(123)
n = 100
a_seq = sin.(linspace(0, 5*pi, n)) + 2 + 0.1 * randn(n)


# == Model parameters == #
γ = 0.8
m = 1
d = γ * [1, -1]
h = 1.0


# == Initial conditions == #
y_m = [2.0]

testlq = LQFilter(d, h, y_m)
y_hist, L, U, y = optimal_y(testlq, a_seq)
y = y[end:-1:1]  # reverse y


# == Plot simulation results == #
fig, ax = subplots(figsize=(10, 6.1))
ax[:set_xlabel]("Time")

# == Some fancy plotting stuff -- simplify if you prefer == #
bbox = (0., 1.01, 1., .101)

time = 1:length(y)
ax[:set_xlim](0, maximum(time))
ax[:plot](time, a_seq / h, "k-o", ms=4, lw=2, alpha=0.6, label=L"$a_t$")
ax[:plot](time, y, "b-o", ms=4, lw=2, alpha=0.6, label=L"$y_t$")
ax[:grid]()
ax[:legend](ncol=2, bbox_to_anchor= bbox, loc = 3, mode = "expand", fontsize= 16)
