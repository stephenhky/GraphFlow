
# Jon Kleinberg's HITS (Hyperlink-Induced Topic Search) algorithm

import numpy as np
from graphflow import L1norm


def hits(adjMatrix, nodes, eps=1e-4, maxstep=1000):
    # hub vector
    i = np.random.uniform(size=len(nodes)).reshape((len(nodes), 1))
    # authority vector
    p = np.random.uniform(size=len(nodes)).reshape((len(nodes), 1))

    step = 0
    converged = False

    while step < maxstep and not converged:
        newp = np.matmul(adjMatrix.T, i)
        newi = np.matmul(adjMatrix, p)

        newp = newp / np.linalg.norm(newp)
        newi = newi / np.linalg.norm(newi)

        converged = (L1norm(newp, p) < eps) and (L1norm(newi, i) < eps)
        i, p = newi, newp

    return i, p
