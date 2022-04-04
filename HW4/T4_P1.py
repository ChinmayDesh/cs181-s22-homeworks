import numpy as np
from matplotlib import pyplot as plt

neg_data = np.array([[-1, -1], [1, -1]])
pos_data = np.array([[-3, 1], [-2, 1], [0, 1], [2, 1], [3, 1]])
pos_x = pos_data[:,0]
neg_x = neg_data[:,0]

pos_x = np.vstack((pos_x, -(8/3)*pos_x**2 + (2/3)*pos_x**4))
neg_x = np.vstack((neg_x, -(8/3)*neg_x**2 + (2/3)*neg_x**4))

plt.scatter(pos_x[0], pos_x[1])
plt.scatter(neg_x[0], neg_x[1])
plt.axhline(y=-1)
plt.show()