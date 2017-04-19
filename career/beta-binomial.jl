#=
Illustrates the usage of the BetaBinomial type

@author : Spencer Lyon <spencer.lyon@nyu.edu>
          Victoria Gregory <victoria.gregory@nyu.edu>

@date: 2014-08-04

=#

using QuantEcon
using Plots
pyplot()
using LaTeXStrings

n = 50
a_vals = [0.5, 1, 100]
b_vals = [0.5, 1, 100]

pdfs = []
labels = []
for (a, b) in zip(a_vals, b_vals)
    d = BetaBinomial(n, a, b)
    push!(pdfs, pdf(d))
    ab_label = LaTeXString("\$a=$a\$, \$b=$b\$")
    push!(labels, ab_label)
end

plot(0:n, pdfs, label=reshape(labels,1,length(labels)), markershape=:circle)
