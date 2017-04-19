type SimpleOG
    B :: Int64
    M :: Int64
    alpha :: Float64
    beta :: Float64
    R :: Array{Float64}
    Q :: Array{Float64}
end

function SimpleOG(;B=10, M=5, alpha=0.5, beta=0.9)

    u(c) = c^alpha
    n = B + M + 1
    m = M + 1

    R = Array{Float64}(n,m)
    Q = zeros(Float64,n,m,n)

    for a in 0:M
        Q[:, a + 1, (a:(a + B)) + 1] = 1 / (B + 1)
        for s in 0:(B + M)
            R[s + 1, a + 1] = a<=s ? u(s - a) : -Inf
        end
    end
    
    return SimpleOG(B, M, alpha, beta, R, Q)
end
