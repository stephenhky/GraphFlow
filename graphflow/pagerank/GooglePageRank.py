import warnings

import numpy as np

from .cpagerank import pagerank_cython
from .. import L1norm, PageRankLanguage


def GoogleMatrix(digraph, beta):
    nodedict = {node: idx for idx, node in enumerate(digraph.nodes())}
    A = (1 - beta) / float(len(digraph)) * np.ones(shape=(len(digraph), len(digraph)))
    for node1, node2 in digraph.edges():
        A[nodedict[node2], nodedict[node1]] += beta / float(len(list(digraph.successors(node1))))
    return A, nodedict


def CalculatePageRankFromAdjacencyMatrix_Cython(adjMatrix, nodes, eps=1e-4, maxstep=1000):
    return pagerank_cython(adjMatrix, nodes, eps, maxstep)


def CalculatePageRankFromAdjacencyMatrix_Python(adjMatrix, nodes, eps=1e-4, maxstep=1000):
    nbnodes = adjMatrix.shape[0]
    r = np.transpose([np.repeat(1 / float(nbnodes), nbnodes)])
    converged = False
    stepid = 0
    while not converged and stepid < maxstep:
        newr = np.matmul(adjMatrix, r)
        converged = (L1norm(newr, r) < eps)
        r = newr
        stepid += 1
    nodepr = {node: r[nodes[node], 0] for node in nodes}
    return nodepr


def CalculatePageRankFromAdjacencyMatrix(adjMatrix, nodes, eps=1e-4, maxstep=1000,
                                         language=PageRankLanguage.CYTHON,
                                         fortran=True):
    if not fortran:
        warnings.warn('The boolean variable "fortran" is deprecated.')
    if language == PageRankLanguage.FORTRAN:
        raise ValueError('Fortran is no longer supported.')
    elif language == PageRankLanguage.CYTHON:
        return CalculatePageRankFromAdjacencyMatrix_Cython(adjMatrix, nodes, eps=eps, maxstep=maxstep)
    else:
        return CalculatePageRankFromAdjacencyMatrix_Python(adjMatrix, nodes, eps=eps, maxstep=maxstep)


def CalculatePageRank(digraph, beta, eps=1e-4, maxstep=1000, fortran=True):
    A, nodes = GoogleMatrix(digraph, beta)
    return CalculatePageRankFromAdjacencyMatrix(A, nodes, eps=eps, maxstep=maxstep, fortran=fortran)

