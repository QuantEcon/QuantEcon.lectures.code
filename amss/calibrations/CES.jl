#=

CES.jl

@author: Shunsuke Hori

=#

module CES

using LS

function baseline(;
    beta = 0.9,
    sigma = 2.0,
    gamma = 2.0,
    Pi = 0.5 *ones(2,2),
    G = [0.1, 0.2],
    Theta = ones(Float64, 2),
    transfers = false
    )
    function U(c,n)
        if sigma == 1.0
            U = log(c)
        else
            U = (c.^(1.0-sigma)-1.0)/(1.0-sigma)
        end
        return U - n.^(1+gamma)/(1+gamma)
    end
    #derivatives of utiltiy function
    Uc(c,n) =  c.^(-sigma)
    Ucc(c,n) = -sigma*c.^(-sigma-1.0)
    Un(c,n) = -n.^gamma
    Unn(c,n) = -gamma*n.^(gamma-1.0)
    n_less_than_one = false
    return LS.Para(beta, Pi, G, Theta, transfers,
                    U, Uc, Ucc, Un, Unn, n_less_than_one)
end

#Model 1
M1 = baseline()

#Model 2

M2 = baseline(G=[0.15], Pi=ones(1,1), Theta=[1.0])

#Model 3 with time varying

M_time_example = baseline(G=[0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
                            Theta = ones(6)# Theta can in principle be random
                            )

M_time_example.Pi = [0.0 1.0 0.0 0.0 0.0 0.0;
                     0.0 0.0 1.0 0.0 0.0 0.0;
                     0.0 0.0 0.0 0.5 0.5 0.0;
                     0.0 0.0 0.0 0.0 0.0 1.0;
                     0.0 0.0 0.0 0.0 0.0 1.0;
                     0.0 0.0 0.0 0.0 0.0 1.0]
end