import numpy as np
import matplotlib.pyplot as plt

x = [np.random.randn() for i in range(100)]
plt.plot(x, 'b-', label="white noise")
plt.legend()
plt.show()
