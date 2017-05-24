
from scipy.stats import norm
from scipy.optimize import brentq

# Some global variables that will stay constant
alpha  = 0.013               
alpha_q = (1-(1-alpha)**3)   # quarterly (alpha is monthly)
b      = 0.0124
d      = 0.00822
beta   = 0.98       
gamma  = 1.0
sigma  = 2.0
    
# The default wage distribution --- a discretized lognormal
log_wage_mean, wage_grid_size, max_wage = 20, 200, 170
logw_dist = norm(np.log(log_wage_mean), 1)
w_vec = np.linspace(0, max_wage, wage_grid_size + 1) 
cdf = logw_dist.cdf(np.log(w_vec))
pdf = cdf[1:]-cdf[:-1]
p_vec = pdf / pdf.sum()
w_vec = (w_vec[1:] + w_vec[:-1])/2


def compute_optimal_quantities(c, tau):
    """
    Compute the reservation wage, job finding rate and value functions of the
    workers given c and tau.

    """
    
    mcm = McCallModel(alpha=alpha_q, 
                     beta=beta, 
                     gamma=gamma, 
                     c=c-tau,         # post tax compensation
                     sigma=sigma, 
                     w_vec=w_vec-tau, # post tax wages
                     p_vec=p_vec)

    w_bar, V, U = compute_reservation_wage(mcm, return_values=True)
    lmda = gamma * np.sum(p_vec[w_vec-tau > w_bar])
    return w_bar, lmda, V, U

def compute_steady_state_quantities(c, tau):
    """
    Compute the steady state unemployment rate given c and tau using optimal
    quantities from the McCall model and computing corresponding steady state
    quantities

    """
    w_bar, lmda, V, U = compute_optimal_quantities(c, tau)
    
    # Compute steady state employment and unemployment rates
    lm = LakeModel(alpha=alpha_q, lmda=lmda, b=b, d=d) 
    x = lm.rate_steady_state()
    e, u = x
    
    # Compute steady state welfare
    w = np.sum(V * p_vec * (w_vec - tau > w_bar)) / np.sum(p_vec * (w_vec -
        tau > w_bar))
    welfare = e * w + u * U
    
    return e, u, welfare


def find_balanced_budget_tax(c):
    """
    Find tax level that will induce a balanced budget.

    """
    def steady_state_budget(t):
        e, u, w = compute_steady_state_quantities(c, t)
        return t - u * c

    tau = brentq(steady_state_budget, 0.0, 0.9 * c)
    return tau



if __name__ == '__main__':

    # Levels of unemployment insurance we wish to study
    c_vec = np.linspace(5, 140, 60)

    tax_vec = []
    unempl_vec = []
    empl_vec = []
    welfare_vec = []

    for c in c_vec:
        t = find_balanced_budget_tax(c)
        e_rate, u_rate, welfare = compute_steady_state_quantities(c, t)
        tax_vec.append(t)
        unempl_vec.append(u_rate)
        empl_vec.append(e_rate)
        welfare_vec.append(welfare)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(c_vec, unempl_vec, 'b-', lw=2, alpha=0.7)
    ax.set_title('unemployment')

    ax = axes[0, 1]
    ax.plot(c_vec, empl_vec, 'b-', lw=2, alpha=0.7)
    ax.set_title('employment')

    ax = axes[1, 0]
    ax.plot(c_vec, tax_vec, 'b-', lw=2, alpha=0.7)
    ax.set_title('tax')

    ax = axes[1, 1]
    ax.plot(c_vec, welfare_vec, 'b-', lw=2, alpha=0.7)
    ax.set_title('welfare')

    plt.tight_layout()
    plt.show()
