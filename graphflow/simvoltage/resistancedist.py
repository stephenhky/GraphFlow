

'''
Get the resistance distance matrix of a simple undirected network.
See: http://en.wikipedia.org/wiki/Resistance_distance
'''

import numpy as np
import sparse

default_nodes = ['Stephen', 'Sinnie', 'Elaine']
default_edges = [('Stephen', 'Sinnie'),
                 ('Elaine', 'Sinnie'),
                 ('Elaine', 'Stephen')]

class GraphResistanceDistance:
    """
    Compute the resistance distance matrix of a simple undirected network.
    
    This class calculates the resistance distance between all pairs of nodes
    in an undirected graph using the Moore-Penrose pseudoinverse of the Laplacian matrix.
    Resistance distance is a measure of how well-connected two nodes are in a network,
    considering all possible paths between them.
    
    See: http://en.wikipedia.org/wiki/Resistance_distance
    """
    def __init__(self, nodes=default_nodes, edges=default_edges):
        """
        Initialize the GraphResistanceDistance class.
        
        Parameters
        ----------
        nodes : list, optional
            List of node identifiers. Default is ['Stephen', 'Sinnie', 'Elaine'].
        edges : list of tuples, optional
            List of edges as tuples (node1, node2). Default is
            [('Stephen', 'Sinnie'), ('Elaine', 'Sinnie'), ('Elaine', 'Stephen')].
        """
        self.initializeClass(nodes, edges)
        self.Omega = self.computeResistanceDistance()
        
    def getResistance(self, node1, node2):
        """
        Get the resistance distance between two nodes.
        
        Parameters
        ----------
        node1 : str
            The identifier of the first node.
        node2 : str
            The identifier of the second node.
        
        Returns
        -------
        float
            The resistance distance between the two nodes.
        
        Raises
        ------
        Exception
            If either node is not in the graph.
        """
        if (node1 in self.nodesIdx) and (node2 in self.nodesIdx):
            idx0 = self.nodesIdx[node1]
            idx1 = self.nodesIdx[node2]
            return self.Omega[idx0, idx1]
        else:
            unknown_keys = [node for node in [node1, node2] if not node in self.nodesIdx]
            raise Exception('Unknown key(s): '+' '.join(unknown_keys))
    
    def initializeClass(self, nodes, edges):
        """
        Initialize the class with nodes and edges.
        
        Parameters
        ----------
        nodes : list
            List of node identifiers.
        edges : list of tuples
            List of edges as tuples (node1, node2).
        """
        self.nodes = nodes
        # all edges are unique
        self.edges = list(set([tuple(sorted(edge)) for edge in edges]))
        self.nodesIdx = {self.nodes[idx]: idx for idx in range(len(self.nodes))}

    def calculateDegreeMatrix(self):
        """
        Calculate the degree matrix of the graph.
        
        The degree matrix is a diagonal matrix where each diagonal entry
        represents the degree of the corresponding node (number of edges connected to it).
        
        Returns
        -------
        sparse.DOK
            The degree matrix as a sparse DOK (Dictionary of Keys) matrix.
        """
        Dmatrix = sparse.DOK((len(self.nodes), len(self.nodes)))
        for edge in self.edges:
            for node in edge:
                idx = self.nodesIdx[node]
                Dmatrix[idx, idx] += 1
        return Dmatrix
        
    def calculateAdjacencyMatrix(self):
        """
        Calculate the adjacency matrix of the graph.
        
        The adjacency matrix is a square matrix where the entry at position (i, j)
        is 1 if there is an edge between nodes i and j, and 0 otherwise.
        
        Returns
        -------
        sparse.DOK
            The adjacency matrix as a sparse DOK (Dictionary of Keys) matrix.
        """
        Amatrix = sparse.DOK((len(self.nodes), len(self.nodes)))
        for edge in self.edges:
            idx0 = self.nodesIdx[edge[0]]
            idx1 = self.nodesIdx[edge[1]]
            Amatrix[idx0, idx1] = 1
            Amatrix[idx1, idx0] = 1
        return Amatrix
        
    def computeResistanceDistance(self):
        """
        Compute the resistance distance matrix for the graph.
        
        This method calculates the resistance distance between all pairs of nodes
        using the Moore-Penrose pseudoinverse of the Laplacian matrix.
        
        Returns
        -------
        sparse.DOK
            The resistance distance matrix as a sparse DOK matrix.
        """
        Dmatrix = self.calculateDegreeMatrix()
        Amatrix = self.calculateAdjacencyMatrix()
        Lmatrix = Dmatrix - Amatrix
        Lambda = np.linalg.pinv(Lmatrix.todense())
        Omega = sparse.DOK((len(self.nodes), len(self.nodes)))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                Omega[i, j] = Lambda[i, i] + Lambda[j, j] - 2 * Lambda[i, j]
        return Omega
