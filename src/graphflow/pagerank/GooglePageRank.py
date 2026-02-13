
from typing import Annotated, Literal

import networkx
import numpy as np
from numpy.typing import NDArray
import numba as nb

from .. import L1norm, PageRankLanguage


def GoogleMatrix(
        digraph: networkx.DiGraph,
        beta: float
) -> tuple[Annotated[NDArray[np.float64], Literal["1D Array"]], dict[str, int]]:
    """
    Compute the Google Matrix for a directed graph.
    
    The Google Matrix is used in PageRank calculations and represents the probability
    transition matrix of a random walk on the graph. It incorporates a damping factor
    to handle dangling nodes and ensure convergence.
    
    Parameters
    ----------
    digraph : networkx.DiGraph
        The directed graph for which to compute the Google Matrix.
    beta : float
        The damping factor (between 0 and 1). Typically set to 0.85.
    
    Returns
    -------
    tuple
        A tuple containing:
        - A (numpy.ndarray): The Google Matrix.
        - nodedict (dict): A dictionary mapping node identifiers to their indices.
    """
    nodedict = {node: idx for idx, node in enumerate(digraph.nodes())}
    A = (1 - beta) / float(len(digraph)) * np.ones(shape=(len(digraph), len(digraph)))
    for node1, node2 in digraph.edges():
        A[nodedict[node2], nodedict[node1]] += beta / float(len(list(digraph.successors(node1))))
    return A, nodedict


@nb.njit(nb.float64[:](nb.float64[:, :], float, int))
def _calculate_pagerank_matrix(
        adjMatrix: Annotated[NDArray[np.float64], Literal["2D Array"]],
        eps: float=1e-4,
        maxstep: int=1000
) -> Annotated[NDArray[np.float64], Literal["1D Array"]]:
    nbnodes = adjMatrix.shape[0]
    r = np.transpose([np.repeat(1 / float(nbnodes), nbnodes)])
    converged = False
    stepid = 0
    while not converged and stepid < maxstep:
        newr = np.matmul(adjMatrix, r)
        converged = (L1norm(newr, r) < eps)
        r = newr
        stepid += 1
    return r


def CalculatePageRankFromAdjacencyMatrix_Python(
        adjMatrix: Annotated[NDArray[np.float64], Literal["2D Array"]],
        nodes: dict[str, int],
        eps: float=1e-4,
        maxstep: int=1000
) -> dict[str, float]:
    """
    Calculate PageRank from an adjacency matrix using Python implementation.
    
    This function computes the PageRank scores for nodes in a graph represented by
    an adjacency matrix. It uses an iterative approach until convergence or maximum
    steps are reached.
    
    Parameters
    ----------
    adjMatrix : numpy.ndarray
        The adjacency matrix representing the graph structure.
    nodes : dict
        A dictionary mapping node identifiers to their indices.
    eps : float, optional
        The convergence threshold. The algorithm stops when the change in vectors
        is less than this value. Default is 1e-4.
    maxstep : int, optional
        The maximum number of iterations to perform. Default is 1000.
    
    Returns
    -------
    dict
        A dictionary mapping node identifiers to their PageRank scores.
    """
    r = _calculate_pagerank_matrix(adjMatrix, eps, maxstep)
    nodepr = {node: r[nodes[node], 0] for node in nodes}
    return nodepr


def CalculatePageRankFromAdjacencyMatrix(
        adjMatrix: Annotated[NDArray[np.float64], Literal["2D Array"]],
        nodes: dict[str, int],
        eps: float=1e-4,
        maxstep: int=1000,
        language: PageRankLanguage=PageRankLanguage.PYTHON
) -> dict[str, float]:
    """
    Calculate PageRank from an adjacency matrix using specified implementation language.
    
    This function computes the PageRank scores for nodes in a graph represented by
    an adjacency matrix. It can use either Cython or Python implementation based on
    the language parameter.
    
    Parameters
    ----------
    adjMatrix : numpy.ndarray
        The adjacency matrix representing the graph structure.
    nodes : dict
        A dictionary mapping node identifiers to their indices.
    eps : float, optional
        The convergence threshold. The algorithm stops when the change in vectors
        is less than this value. Default is 1e-4.
    maxstep : int, optional
        The maximum number of iterations to perform. Default is 1000.
    language : PageRankLanguage, optional
        The implementation language to use. Default is PageRankLanguage.PYTHON.
    
    Returns
    -------
    dict
        A dictionary mapping node identifiers to their PageRank scores.
    
    """
    if language == PageRankLanguage.PYTHON:
        return CalculatePageRankFromAdjacencyMatrix_Python(adjMatrix, nodes, eps=eps, maxstep=maxstep)
    else:
        raise ValueError("Only Python implemented.")


def CalculatePageRank(
        digraph: networkx.DiGraph,
        beta: float,
        eps: float=1e-4,
        maxstep: int=1000
) -> dict[str, float]:
    """
    Calculate PageRank for a directed graph.
    
    This function computes the PageRank scores for nodes in a directed graph.
    It first computes the Google Matrix and then calculates the PageRank using
    the specified parameters.
    
    Parameters
    ----------
    digraph : networkx.DiGraph
        The directed graph for which to compute PageRank.
    beta : float
        The damping factor (between 0 and 1). Typically set to 0.85.
    eps : float, optional
        The convergence threshold. The algorithm stops when the change in vectors
        is less than this value. Default is 1e-4.
    maxstep : int, optional
        The maximum number of iterations to perform. Default is 1000.

    Returns
    -------
    dict
        A dictionary mapping node identifiers to their PageRank scores.
    """
    A, nodes = GoogleMatrix(digraph, beta)
    return CalculatePageRankFromAdjacencyMatrix(A, nodes, eps=eps, maxstep=maxstep)

