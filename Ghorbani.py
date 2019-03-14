import numpy as np
from neldermead import NelderMead
import qutip

dim = 3
f = lambda x: np.sum(x**2)
simplex = np.zeros([dim, dim + 1])
for i in range(dim + 1):
    simplex[:, i] = np.array([np.random.rand() for _ in range(dim)])
nm = NelderMead(dim, f, simplex)

x_best, f_best = nm.optimize(100)
print("x_best:{}, f_best:{}".format(x_best, f_best))

#x_best:[[-1.48045204e-08]
# [-1.80962770e-08]
# [ 5.08040874e-08]], f_best:3.1277043680572982e-15