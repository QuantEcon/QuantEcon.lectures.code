#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>

=#

# == Parameters == #
bet = 1 / 1.05
rho, mg = .7, .35
A = eye(2)
A = [rho mg*(1 - rho); 0.0 1.0]
C = [sqrt(1 - rho^2)*mg/10 0.0;0 0]
Sg = [1.0 0.0]
Sd = [0.0 0.0]
Sb = [0 2.135]
Ss = [0.0 0.0]
discrete = false
proc = ContStochProcess(A, C)

econ = Economy(bet, Sg, Sd, Sb, Ss, discrete, proc)
T = 50

path = compute_paths(econ, T)

gen_fig_1(path)
