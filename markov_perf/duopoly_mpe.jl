
#=
@authors: Shunsuke Hori

Markov Perfect Equilibrium for the simple duopoly example.

See the lecture at http://quant-econ.net/jl/markov_perf.html for a
description of the model.
=#

using QuantEcon

# == Parameters == #
a0    = 10.0
a1    = 2.0
beta  = 0.96
gamma = 12.0

# == In LQ form == #

A  = eye(3)

B1 = [0.0, 1.0, 0.0]
B2 = [0.0, 0.0, 1.0]


R1 = [0.0      -a0/2.0 0.0;
      -a0/2.0  a1      a1/2.0;
      0.0      a1/2.0  0.0]

R2 = [0.0     0.0   -a0/2.0;
      0.0     0.0    a1/2.0;
      -a0/2.0 a1/2.0 a1]

Q1 = Q2 = gamma

S1 = S2 = W1 = W2 = M1 = M2 = 0.0

# == Solve using QE's nnash function == #
F1, F2, P1, P2 = nnash(A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2,
                          beta=beta)

# == Display policies == #
println("Computed policies for firm 1 and firm 2:")
println("F1 = $F1")
println("F2 = $F2")
