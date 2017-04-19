
import matplotlib.pyplot as plt
from lucastree import *

tree = LucasTree()

f = np.zeros(len(tree.grid))
fig, ax = plt.subplots()

for i in range(1, 200, 10):
    f = lucas_operator(f, tree)
    ax.plot(tree.grid, f)

plt.show()

