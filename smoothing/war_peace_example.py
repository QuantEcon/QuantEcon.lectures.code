import numpy as np

# Parameters

beta = .96
y = [1, 2]
b0 = 0
P = np.asarray([[.8, .2],
                [.4, .6]])

cp = ConsumptionProblem(beta, y, b0, P)
Q = beta*P
N_simul = 150

c_bar, b1, b2 = consumption_complete(cp)
debt_complete = np.asarray([b1, b2])

print("P = ", P)
print("Q= ", Q, "\n")
print("Govt expenditures in peace and war =", y)
print("Constant tax collections = ", c_bar)
print("Govt assets in two states = ", debt_complete)

msg = """
Now let's check the government's budget constraint in peace and war.
Our assumptions imply that the government always purchases 0 units of the
Arrow peace security.
"""
print(msg)

AS1 = Q[0,1] * b2
print("Spending on Arrow war security in peace = ", AS1)
AS2 = Q[1,1]*b2
print("Spending on Arrow war security in war = ", AS2)

print("\n")
print("Government tax collections plus asset levels in peace and war")
TB1=c_bar+b1
print("T+b in peace = ",TB1 )
TB2 = c_bar + b2
print("T+b in war = ", TB2)


print("\n")
print("Total government spending in peace and war")
G1= y[0] + AS1
G2 = y[1] + AS2
print("total govt spending in peace = ", G1)
print("total govt spending in war = ", G2)


print("\n")
print("Let's see ex post and ex ante returns on Arrow securities")

Pi= np.reciprocal(Q)
exret= Pi
print("Ex post returns to purchase of Arrow securities = ", exret)
exant = Pi*P
print("Ex ante returns to purchase of Arrow securities ", exant)
