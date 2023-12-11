

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

    """
    def __init__(self, nodes=default_nodes, edges=default_edges):
        """

        :param nodes:
        :param edges:
        """
        self.initializeClass(nodes, edges)
        self.Omega = self.computeResistanceDistance()
        
    def getResistance(self, node1, node2):
        """

        :param node1:
        :param node2:
        :return:
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

        :param nodes:
        :param edges:
        :return:
        """
        self.nodes = nodes
        # all edges are unique
        self.edges = list(set([tuple(sorted(edge)) for edge in edges]))
        self.nodesIdx = {self.nodes[idx]: idx for idx in range(len(self.nodes))}

    def calculateDegreeMatrix(self):
        """

        :return:
        """
        Dmatrix = sparse.DOK((len(self.nodes), len(self.nodes)))
        for edge in self.edges:
            for node in edge:
                idx = self.nodesIdx[node]
                Dmatrix[idx, idx] += 1
        return Dmatrix
        
    def calculateAdjacencyMatrix(self):
        """

        :return:
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

        :return:
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
