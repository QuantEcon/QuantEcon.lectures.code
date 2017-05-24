

srand(1)
N_simul = 150
cp = ConsumptionProblem()

c_bar, b1, b2 = consumption_complete(cp)
debt_complete = [b1, b2]

c_path, debt_path, y_path, s_path = consumption_incomplete(cp, N_simul=N_simul)

fig, ax = subplots(1, 2, figsize = (15, 5))

ax[1][:set_title]("Tax collection paths", fontsize = 17)
ax[1][:plot](1:N_simul, c_path, label = "incomplete market", lw = 3)
ax[1][:plot](1:N_simul, c_bar * ones(N_simul), label = "complete market", lw = 3)
ax[1][:plot](1:N_simul, y_path, label = "govt expenditures", lw = 2, 
           alpha = .6, linestyle = "--")
ax[1][:legend](loc = "best", fontsize = 15)
ax[1][:set_xlabel]("Periods", fontsize = 13)
ax[1][:set_ylim]([1.4, 2.1])

ax[2][:set_title]("Government assets paths", fontsize = 17)
ax[2][:plot](1:N_simul, debt_path, label = "incomplete market", lw = 2)
ax[2][:plot](1:N_simul, debt_complete[s_path], label = "complete market", lw = 2)
ax[2][:plot](1:N_simul, y_path, label = "govt expenditures", lw = 2, 
           alpha = .6, linestyle = "--")
ax[2][:legend](loc = "best", fontsize = 15)
ax[2][:axhline](0, color = "k", lw = 1)
ax[2][:set_xlabel]("Periods", fontsize = 13)
