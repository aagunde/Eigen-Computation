#SVD and Eigen Value Decomposition

#import scipy, numpy library
from scipy import linalg
import numpy as np

#initialize matrix
M = np.matrix([[1, 2], [2, 1], [3, 4], [4, 3]])

#decompose matrix M into SVD components
U, sig, Vt = linalg.svd(M, full_matrices=False)

#eigenvalue decomposition
Evals, Evecs = linalg.eigh(M.transpose()* M)

#sorting
sortIndex = Evals.argsort()[::-1]
Evals = Evals[sortIndex]
Evecs = Evecs[:,sortIndex]

#display results
print(U)
print(sig)
print(Vt)
print(Evals)
print(Evecs)
