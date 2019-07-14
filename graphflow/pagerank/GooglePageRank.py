import numpy as np
#import networkx as nx

from f90pagerank import f90pagerank as fpr

from graphflow import L1norm

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
    A = np.matrix((1 - beta) / float(len(digraph)) * np.ones(shape=(len(digraph), len(digraph))))
    for node1, node2 in digraph.edges():
        A[nodedict[node2], nodedict[node1]] += beta / float(len(list(digraph.successors(node1))))
    return A, nodedict


def CalculatePageRankFromAdjacencyMatrix(adjMatrix, nodes, eps=1e-4, maxstep=1000, fortran=True):
    """

    :param adjMatrix:
    :param nodes:
    :param eps:
    :param maxstep:
    :param fortran:
    :return:
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


def CalculatePageRank(digraph, beta, eps=1e-4, maxstep=1000, fortran=True):
    A, nodes = GoogleMatrix(digraph, beta)
    return CalculatePageRankFromAdjacencyMatrix(A, nodes, eps=eps, maxstep=maxstep, fortran=fortran)
