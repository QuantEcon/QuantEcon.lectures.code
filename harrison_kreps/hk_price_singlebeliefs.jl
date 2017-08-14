#=

Authors: Shunsuke Hori

=#
"""
Function to Solve Single Beliefs
"""
function price_singlebeliefs(transition::Matrix, dividend_payoff::Vector;
                             beta::AbstractFloat=.75)
    # First compute inverse piece
    imbq_inv = inv(eye(size(transition,1)) - beta*transition)

    # Next compute prices
    prices = beta * ((imbq_inv*transition)* dividend_payoff)

    return prices
end
