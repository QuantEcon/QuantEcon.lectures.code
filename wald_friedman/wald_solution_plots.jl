# Choose parameters
c = 1.25
L0 = 27.0
L1 = 27.0

# Choose n points and distributions
m = 251
f0 = pdf(Beta(2.5, 3), linspace(0, 1, m))
f0 = f0 / sum(f0)
f1 = pdf(Beta(3, 2.5), linspace(0, 1, m))
f1 = f1 / sum(f1)  # Make sure sums to 1

# Create an instance of our WaldFriedman class
wf = WaldFriedman(c, L0, L1, f0, f1; m=m);


# Solve and simulate the solution
cdist, tdist = stopping_dist(wf; ndraws=5000)


a = plot([f0 f1], 
    xlabel=L"$k$ Values",
    ylabel=L"Probability of $z_k$",
    labels=reshape([L"$f_0$"; L"$f_1$"], 1, 2),
    linewidth=2,
    title="Distributions over Outcomes")

b = plot(wf.pgrid, wf.sol.J, 
    xlabel=L"$p_k$",
    ylabel="Value of Bellman",
    linewidth=2,
    title="Bellman Equation")
    plot!(fill(wf.sol.lb, 2), [minimum(wf.sol.J); maximum(wf.sol.J)],
    linewidth=2, color=:black, linestyle=:dash, label="", ann=(wf.sol.lb-0.05, 5., L"\beta"))
    plot!(fill(wf.sol.ub, 2), [minimum(wf.sol.J); maximum(wf.sol.J)],
    linewidth=2, color=:black, linestyle=:dash, label="", ann=(wf.sol.ub+0.02, 5., L"\alpha"),
    legend=:none)

counts = Array(Int64, maximum(tdist))
for i in 1:maximum(tdist)
    counts[i] = sum(tdist .== i)
end
c = bar(counts,
    xticks=0:1:maximum(tdist),
    xlabel="Time",
    ylabel="Frequency",
    title="Stopping Times",
    legend=:none)

counts = Array(Int64, 2)
for i in 1:2
    counts[i] = sum(cdist .== i-1)
end
d = bar([0; 1],
    counts, 
    xticks=[0; 1],
    title="Correct Decisions", 
    ann=(-.4, 0.6 * sum(cdist), "Percent Correct = $(sum(cdist)/length(cdist))"),
    legend=:none)

plot(a, b, c, d, layout=(2, 2), size=(1200, 800))