
# Jon Kleinberg's HITS (Hyperlink-Induced Topic Search) algorithm

import networkx
import numpy as np
import networkx as nx
from nptyping import NDArray, Shape, Float

from .. import L1norm


def hits(
        adjMatrix: NDArray[Shape["*, *"], Float],
        eps: float=1e-4,
        maxstep: int=1000
) -> tuple[NDArray[Shape["*"], Float], NDArray[Shape["*"], Float]]:
    """
    Compute the HITS (Hyperlink-Induced Topic Search) algorithm on an adjacency matrix.
    
    This function calculates the hub and authority vectors for a given adjacency matrix
    using the HITS algorithm. The algorithm iteratively computes these vectors until
    convergence or the maximum number of steps is reached.
    
    Parameters
    ----------
    adjMatrix : numpy.ndarray
        The adjacency matrix representing the graph structure.
    eps : float, optional
        The convergence threshold. The algorithm stops when the change in vectors
        is less than this value. Default is 1e-4.
    maxstep : int, optional
        The maximum number of iterations to perform. Default is 1000.
    
    Returns
    -------
    tuple
        A tuple containing:
        - hub vector (numpy.ndarray): The hub scores for each node.
        - authority vector (numpy.ndarray): The authority scores for each node.
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


def CalculateHITS(
        digraph: networkx.DiGraph,
        eps: float=1e-4,
        maxstep: int=1000
) -> tuple[NDArray[Shape["*"], Float], NDArray[Shape["*"], Float]]:
    """
    Compute the HITS (Hyperlink-Induced Topic Search) algorithm on a NetworkX digraph.
    
    This function calculates the hub and authority scores for each node in a directed graph
    using the HITS algorithm. It converts the graph to an adjacency matrix and then applies
    the HITS algorithm.
    
    Parameters
    ----------
    digraph : networkx.DiGraph
        The directed graph on which to compute the HITS algorithm.
    eps : float, optional
        The convergence threshold. The algorithm stops when the change in vectors
        is less than this value. Default is 1e-4.
    maxstep : int, optional
        The maximum number of iterations to perform. Default is 1000.
    
    Returns
    -------
    tuple
        A tuple containing:
        - hubdict (dict): A dictionary mapping node identifiers to their hub scores.
        - authdict (dict): A dictionary mapping node identifiers to their authority scores.
    """
    A = nx.adjacency_matrix(digraph).toarray()
    nodes = list(digraph.nodes())
    hubvec, authvec = hits(A, eps=eps, maxstep=maxstep)
    hubdict = {nodes[i]: hubvec[i] for i in range(len(hubvec))}
    authdict = {nodes[i]: authvec[i] for i in range(len(authvec))}
    return hubdict, authdict
