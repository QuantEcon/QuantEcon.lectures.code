#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

=#

using Plots
pyplot()
using Distributions
using LaTeXStrings

srand(42)  # reproducible results
ns = [1, 2, 4, 8]
dom = 0:9

pdfs = []
titles = []
for n in ns
    b = Binomial(n, 0.5)
    push!(pdfs, pdf(b, dom))
    t = LaTeXString("\$n = $n\$")
    push!(titles, t)
end

bar(dom, pdfs, layout=4, alpha=0.6, xlims=(-0.5, 8.5), ylims=(0, 0.55),
    xticks=dom, yticks=[0.0, 0.2, 0.4], legend=:none, title=reshape(titles, 1, length(titles)))
