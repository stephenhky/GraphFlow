
# Jon Kleinberg's HITS (Hyperlink-Induced Topic Search) algorithm

import numpy as np
import networkx as nx
from graphflow import L1norm


def hits(adjMatrix, eps=1e-4, maxstep=1000):
    """

    :param adjMatrix:
    :param eps:
    :param maxstep:
    :return:
    """
    nbnodes = adjMatrix.shape[0]
    # hub vector
    i = np.random.uniform(size=nbnodes).reshape((nbnodes, 1))
    # authority vector
    p = np.random.uniform(size=nbnodes).reshape((nbnodes, 1))

    step = 0
    converged = False

    while step < maxstep and not converged:
        newp = np.matmul(adjMatrix.T, i)
        newi = np.matmul(adjMatrix, p)

        newp = newp / np.linalg.norm(newp)
        newi = newi / np.linalg.norm(newi)

        converged = (L1norm(newp, p) < eps) and (L1norm(newi, i) < eps)
        i, p = newi, newp

    return i.reshape(nbnodes), p.reshape(nbnodes)


def CalculateHITS(digraph, eps=1e-4, maxstep=1000):
    """

    :param digraph:
    :param eps:
    :param maxstep:
    :return:
    """
    A = nx.adjacency_matrix(digraph).toarray()
    nodes = list(digraph.nodes())
    hubvec, authvec = hits(A, eps=eps, maxstep=maxstep)
    hubdict = {nodes[i]: hubvec[i] for i in range(len(hubvec))}
    authdict = {nodes[i]: authvec[i] for i in range(len(authvec))}
    return hubdict, authdict
