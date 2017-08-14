alpha = 0.4
beta = 0.96
mu = 0
s = 0.1

ab = alpha * beta
c1 = log(1 - ab) / (1 - beta)
c2 = (mu + alpha * log(ab)) / (1 - alpha)
c3 = 1 / (1 - beta)
c4 = 1 / (1 - ab)

# Utility 
u(c) = log(c)

u_prime(c) = 1 / c

# Deterministic part of production function
f(k) = k^alpha

f_prime(k) = alpha * k^(alpha - 1)

# True optimal policy
c_star(y) = (1 - alpha * beta) * y

# True value function
v_star(y) = c1 + c2 * (c3 - c4) + c4 * log(y)
    


