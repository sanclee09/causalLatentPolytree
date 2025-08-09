import numpy as np
from latent_polytree import separation, tree, polytree

# example discrepancy matrix Γ_V from Example 12
gamma_ex12 = np.array([
     [0, 2, 3, 1, 3, 4],
     [0, 0, -2, 0, -1, 1],
     [1, -3, 0, 1, 1, -3],
     [0, 1, 2, 0, 2, 3],
     [0, -1, 0, 0, 0, -2],
     [0, 0, -1, 0, -1, 0],
 ])
sibling_groups = separation(gamma_ex12)
T = tree(gamma_ex12)
T.edges

print(T.edges)