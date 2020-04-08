
import numpy as np
cimport numpy as np


def pagerank_cython(np.array adjMatrix, dict nodes, float eps, int maxstep):
    cdef int nbnodes = adjMatrix.shape[0]
    cdef np.array r = np.transpose([np.repeat(1 / float(nbnodes), nbnodes)])
    cdef np.array newr = np.transpose([np.repeat(1 / float(nbnodes), nbnodes)])
    cdef bool converged = False
    cdef int stepid = 0

    while not converged and stepid < maxstep:
        newr = np.matmul(adjMatrix, r)
        converged = np.sum(np.abs(newr - r)) < eps
        r = newr
        stepid += 1

    cdef dict nodepr = {node: r[nodes[node], 0] for node in nodes}

    return nodepr