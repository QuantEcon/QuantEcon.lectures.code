#=

@author : Victoria Gregory, John Stachurski

@date: 2016-08-03

Provides a class that simulates the dynamics of unemployment and employment in
the lake model.

=#


type LakeModel
    lambda :: Float64
    alpha :: Float64
    b :: Float64
    d :: Float64
    g :: Float64
    A :: Matrix{Float64}
    A_hat :: Matrix{Float64}
end

"""
Constructor with default values for `LakeModel`

##### Fields of `LakeModel`

  - lambda : job finding rate
  - alpha : dismissal rate
  - b : entry rate into labor force
  - d : exit rate from labor force
  - g : net entry rate
  - A : updates stock
  - A_hat : updates rate

"""
function LakeModel(;lambda=0.283, alpha=0.013, b=0.0124, d=0.00822)

    g = b - d
    A = [ (1-d) * (1-alpha)  (1-d) * lambda;
          (1-d) * alpha + b (1-lambda) * (1-d) + b]
    A_hat = A ./ (1 + g)

    return LakeModel(lambda, alpha, b, d, g, A, A_hat)
end

"""
Finds the steady state of the system :math:`x_{t+1} = \hat A x_{t}`

##### Arguments

  - lm : instance of `LakeModel`
  - tol: convergence tolerance

##### Returns

  - x : steady state vector of employment and unemployment rates

"""

function rate_steady_state(lm::LakeModel, tol=1e-6)
    x = 0.5 * ones(2)
    error = tol + 1
    while (error > tol)
        new_x = lm.A_hat * x
        error = maxabs(new_x - x)
        x = new_x
    end
    return x
end

"""
Simulates the the sequence of Employment and Unemployent stocks

##### Arguments

  - X0 : contains initial values (E0, U0)
  - T : number of periods to simulate

##### Returns

  - X_path : contains sequence of employment and unemployment stocks

"""

function simulate_stock_path(lm::LakeModel, X0::Vector{Float64}, T::Int)
    X_path = Array(Float64, 2, T)
    X = copy(X0)
    for t in 1:T
        X_path[:, t] = X
        X = lm.A * X
    end
    return X_path
end

"""
Simulates the the sequence of employment and unemployent rates.

##### Arguments

  - X0 : contains initial values (E0, U0)
  - T : number of periods to simulate

##### Returns

  - X_path : contains sequence of employment and unemployment rates

"""
function simulate_rate_path(lm::LakeModel, x0::Vector{Float64}, T::Int)
    x_path = Array(Float64, 2, T)
    x = copy(x0)
    for t in 1:T
        x_path[:, t] = x
        x = lm.A_hat * x
    end
    return x_path
end
    

