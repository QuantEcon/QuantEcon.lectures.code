
mutable struct UncertaintyTrapEcon{TF<:AbstractFloat, TI<:Integer}
    a::TF          # Risk aversion
    gx::TF         # Production shock precision
    rho::TF        # Correlation coefficient for theta
    sig_theta::TF  # Std dev of theta shock
    num_firms::TI      # Number of firms
    sig_F::TF      # Std dev of fixed costs
    c::TF          # External opportunity cost
    mu::TF         # Initial value for mu
    gamma::TF      # Initial value for gamma
    theta::TF      # Initial value for theta
    sd_x::TF       # standard deviation of shock
end

function UncertaintyTrapEcon(;a::AbstractFloat=1.5, gx::AbstractFloat=0.5,
                             rho::AbstractFloat=0.99, sig_theta::AbstractFloat=0.5,
                             num_firms::Integer=100, sig_F::AbstractFloat=1.5,
                             c::AbstractFloat=-420.0, mu_init::AbstractFloat=0.0,
                             gamma_init::AbstractFloat=4.0,
                             theta_init::AbstractFloat=0.0)
    sd_x = sqrt(a/gx)
    UncertaintyTrapEcon(a, gx, rho, sig_theta, num_firms, sig_F, c, mu_init,
                        gamma_init, theta_init, sd_x)

end

function psi(uc::UncertaintyTrapEcon, F::Real)
    temp1 = -uc.a * (uc.mu - F)
    temp2 = 0.5 * uc.a^2 * (1/uc.gamma + 1/uc.gx)
    return (1/uc.a) * (1 - exp(temp1 + temp2)) - uc.c
end

"""
Update beliefs (mu, gamma) based on aggregates X and M.
"""
function update_beliefs!(uc::UncertaintyTrapEcon, X::Real, M::Real)
    # Simplify names
    gx, rho, sig_theta = uc.gx, uc.rho, uc.sig_theta

    # Update mu
    temp1 = rho * (uc.gamma*uc.mu + M*gx*X)
    temp2 = uc.gamma + M*gx
    uc.mu =  temp1 / temp2

    # Update gamma
    uc.gamma = 1 / (rho^2 / (uc.gamma + M * gx) + sig_theta^2)
end

update_theta!(uc::UncertaintyTrapEcon, w::Real) =
    (uc.theta = uc.rho*uc.theta + uc.sig_theta*w)

"""
Generate aggregates based on current beliefs (mu, gamma).  This
is a simulation step that depends on the draws for F.
"""
function gen_aggregates(uc::UncertaintyTrapEcon)
    F_vals = uc.sig_F * randn(uc.num_firms)

    M = sum(psi.(uc, F_vals) .> 0)  # Counts number of active firms
    if M > 0
        x_vals = uc.theta + uc.sd_x * randn(M)
        X = mean(x_vals)
    else
        X = 0.0
    end
    return X, M
end
