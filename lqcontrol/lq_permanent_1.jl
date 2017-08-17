using QuantEcon
using Plots
pyplot()

# == Model parameters == #
r = 0.05
bet = 1 / (1 + r)
T = 45
c_bar = 2.0
sigma = 0.25
mu = 1.0
q = 1e6

# == Formulate as an LQ problem == #
Q = 1.0
R = zeros(2, 2)
Rf = zeros(2, 2); Rf[1, 1] = q
A = [1.0+r -c_bar+mu;
     0.0 1.0]
B = [-1.0; 0.0]
C = [sigma; 0.0]

# == Compute solutions and simulate == #
lq = LQ(Q, R, A, B, C; bet = bet, capT = T, rf = Rf)
x0 = [0.0; 1.0]
xp, up, wp = compute_sequence(lq, x0)

# == Convert back to assets, consumption and income == #
assets = vec(xp[1, :])           # a_t
c = vec(up + c_bar)              # c_t
income = vec(wp[1, 2:end] + mu)  # y_t

# == Plot results == #
p=plot(Vector[assets, c, zeros(T + 1), income, cumsum(income - mu)],
  lab = ["assets" "consumption" "" "non-financial income" "cumulative unanticipated income"],
  color = [:blue :green :black :orange :red],
  width = 3, xaxis = ("Time"), layout = (2, 1),
  bottom_margin = 20mm, size = (600, 600), show = false)


