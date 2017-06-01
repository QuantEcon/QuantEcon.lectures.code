#=

BGP.jl

@author: Shunsuke Hori

=#

## BGP
function baseline_BGP(;
    beta = 0.9,
    psi = 0.69,
    Pi = 0.5 *ones(2,2),
    G = [0.1,0.2],
    Theta = ones(2),
    transfers = false)
    #derivatives of utiltiy function
    U(c,n) = log(c) + psi*log(1-n)
    Uc(c,n) = 1./c
    Ucc(c,n) = -c.^(-2.0)
    Un(c,n) = -psi./(1.0-n)
    Unn(c,n) = -psi./(1.0-n).^2.0
    return Para(beta, Pi, G, Theta, transfers,
                    U, Uc, Ucc, Un, Unn)
end

#Model 1
M1_BGP = baseline_BGP()

#Model 2
M2_BGP = baseline_BGP()
M2_BGP.G = [0.15]
M2_BGP.Pi = ones(1,1)
M2_BGP.Theta = [1.0]
