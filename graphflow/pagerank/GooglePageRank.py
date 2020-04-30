
import enum
import warnings

import numpy as np
#import networkx as nx

from f90pagerank import f90pagerank as fpr
from . import pagerank_cython
from graphflow import L1norm


class PageRankLanguage(enum.Enum):
    PYTHON = 0
    CYTHON = 1
    FORTRAN = 2


def GoogleMatrix(digraph, beta):
    """ Convert the directional graph (networkx) to an adjacency matrix, and the corresponding
    dictionary that maps the name of the nodes to an index.

    :param digraph: directional graph in networkx
    :param beta: probablility of leaking (between 0.0 and 1.0 (inclusive)
    :return: adjacency matrix, dictionary
    :type digraph: networkx.Digraph
    :type beta: float
    :rtype: tuple(numpy.ndarray, dict)
    """
    nodedict = {node: idx for idx, node in enumerate(digraph.nodes())}
    A = (1 - beta) / float(len(digraph)) * np.ones(shape=(len(digraph), len(digraph)))
    for node1, node2 in digraph.edges():
        A[nodedict[node2], nodedict[node1]] += beta / float(len(list(digraph.successors(node1))))
    return A, nodedict


def CalculatePageRankFromAdjacencyMatrix(adjMatrix, nodes, eps=1e-4, maxstep=1000, fortran=True):
    """ From the adjacency matrix, calculate the pagerank for each node.

    :param adjMatrix: the adjacency matrix output from :func:`GoogleMatrix`.
    :param nodes: a list of nodes
    :param eps: tolerated error for convergence (default: 1e-4)
    :param maxstep: maximum number of iterations (default: 1000)
    :param fortran: use Fortran native code or not (default: True)
    :return: a dictionary of pagerank scores for all the nodes
    :type adjMatrix: numpy.ndarray
    :type nodes: list
    :type eps: float
    :type maxstep: int
    :type fortran: bool
    :rtype: dict
    """
    if fortran:
        r = fpr.compute_pagerank(adjMatrix, eps, maxstep)
        nodepr = {node: r[nodes[node]] for node in nodes}
    else:
        nbnodes = adjMatrix.shape[0]
        r = np.transpose(np.matrix(np.repeat(1 / float(nbnodes), nbnodes)))
        converged = False
        stepid = 0
        while not converged and stepid < maxstep:
            newr = adjMatrix * r
            converged = (L1norm(newr, r) < eps)
            r = newr
            stepid += 1
        nodepr = {node: r[nodes[node], 0] for node in nodes}
    return nodepr


def CalculatePageRankFromAdjacencyMatrix(adjMatrix, nodes, eps=1e-4, maxstep=1000,
                                         language=PageRankLanguage.FORTRAN,
                                         fortran=True):
    if not fortran:
        warnings.warn('The boolean variable "fortran" is deprecated.')
    if language == PageRankLanguage.FORTRAN:
        return CalculatePageRankFromAdjacencyMatrix_Fortran(adjMatrix, nodes, eps=eps, maxstep=maxstep)
    elif language == PageRankLanguage.CYTHON:
        return CalculatePageRankFromAdjacencyMatrix_Cython(adjMatrix, nodes, eps=eps, maxstep=maxstep)
    else:
        return CalculatePageRankFromAdjacencyMatrix_Python(adjMatrix, nodes, eps=eps, maxstep=maxstep)


def CalculatePageRank(digraph, beta, eps=1e-4, maxstep=1000, fortran=True):
    """ Given a directional graph, compute the pagerank for all the nodes.

    :param digraph: directional graph in networkx
    :param beta: probablility of leaking (between 0.0 and 1.0 (inclusive)
    :param eps: tolerated error for convergence (default: 1e-4)
    :param maxstep: maximum number of iterations (default: 1000)
    :param fortran: use Fortran native code or not (default: True)
    :return: a dictionary of pagerank scores for all the nodes
    :type digraph: networkx.DiGraph
    :type beta: float
    :type eps: float
    :type maxstep: int
    :type fortran: bool
    :rtype: dict
    """
    A, nodes = GoogleMatrix(digraph, beta)
    return CalculatePageRankFromAdjacencyMatrix(A, nodes, eps=eps, maxstep=maxstep, fortran=fortran)
