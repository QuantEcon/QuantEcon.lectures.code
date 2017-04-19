a = plot([f0 f1], 
    xlabel=L"$k$ Values",
    ylabel=L"Probability of $z_k$",
    labels=reshape([L"$f_0$"; L"$f_1$"], 1, 2),
    linewidth=2,
    ylims=[0.;0.07],
    title="Original Distributions")

mix = Array(Float64, 50, 3)
p_k = [0.25; 0.5; 0.75]
labels = Array(String, 3)
for i in 1:3
    mix[:, i] = p_k[i] * f0 + (1 - p_k[i]) * f1
    labels[i] = string(L"$p_k$ = ", p_k[i])
end
    
b = plot(mix, 
    xlabel=L"$k$ Values",
    ylabel=L"Probability of $z_k$",
    labels=reshape(labels, 1, 3),
    linewidth=2,
    ylims=[0.;0.06],
    title="Mixture of Original Distributions")

plot(a, b, layout=(2, 1), size=(600, 800))

