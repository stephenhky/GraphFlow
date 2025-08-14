
from itertools import product

import networkx as nx


DEFAULT_NODES = ['Stephen', 'Sinnie', 'Elaine']
DEFAULT_EDGES = [('Stephen', 'Sinnie', 0.2),
                 ('Sinnie', 'Stephen', 0.2),
                 ('Sinnie', 'Elaine', 0.3),
                 ('Elaine', 'Sinnie', 0.2),
                 ('Stephen', 'Elaine', 1.1),
                 ('Elaine', 'Stephen', 1.2)]


class SocialNetworkSimVoltage:
    """
    Simulate voltage in a social network to compute resistance distances.
    
    This class models a social network as an electrical circuit where edges have
    resistance values. It computes the effective resistance between any two nodes
    using circuit simulation techniques.
    """
    def __init__(
            self,
            nodes: list[str]=None,
            edges: list[tuple[str, str, float]]=None,
            precalculated_distance: bool=True
    ):
        """
        Initialize the SocialNetworkSimVoltage class.
        
        Parameters
        ----------
        nodes : list[str], optional
            List of node identifiers. Default is ['Stephen', 'Sinnie', 'Elaine'].
        edges : list of tuples, optional
            List of edges as tuples (node1, node2, weight). Default is
            [('Stephen', 'Sinnie', 0.2), ('Sinnie', 'Stephen', 0.2), ('Sinnie', 'Elaine', 0.3),
             ('Elaine', 'Sinnie', 0.2), ('Stephen', 'Elaine', 1.1), ('Elaine', 'Stephen', 1.2)].
        precalculated_distance : bool, optional
            Whether to precalculate distances. Default is True.
        """
        self.initializeClass(nodes, edges)
        self.precalculated_distance = precalculated_distance
        if self.precalculated_distance:
            self.precalculate_distance()

    def initializeClass(self, nodes: list[str], edges: list[tuple[str, str, float]]) -> None:
        """
        Initialize the class with nodes and edges.
        
        Parameters
        ----------
        nodes : list[str]
            List of node identifiers.
        edges : list of tuples
            List of edges as tuples (node1, node2, weight).
        """
        self.constructSocialNetwork(nodes, edges)
        self.errTol = 1e-4
        self.maxSteps = 10000

    def precalculate_distance(self) -> dict[tuple[str, str], float]:
        """
        Precalculate the shortest path distances between all pairs of nodes.
        
        This method computes the shortest path length between all pairs of nodes
        in the network and stores them in a dictionary for quick access.
        If no path exists between two nodes, the distance is set to infinity.
        """
        self.distance_matrix = {}
        for person1, person2 in product(self.wordNet.nodes(), self.wordNet.nodes()):
            try:
                self.distance_matrix[(person1, person2)] = float(nx.shortest_path_length(self.wordNet, person1, person2, weight='weight'))
            except nx.exception.NetworkXNoPath:
                self.distance_matrix[(person1, person2)] = float('inf')

    def constructSocialNetwork(self, nodes: list[str], edges: list[tuple[str, str, float]]) -> None:
        """
        Construct the social network as a directed graph.
        
        This method creates a NetworkX DiGraph from the provided nodes and edges.
        
        Parameters
        ----------
        nodes : list[str]
            List of node identifiers.
        edges : list of tuples
            List of edges as tuples (node1, node2, weight).
        """
        self.wordNet = nx.DiGraph()
        self.wordNet.add_nodes_from(nodes)
        self.wordNet.add_weighted_edges_from(edges)
        
    def checkPersonIrrelevant(self, person: str, person1: str, person2: str) -> bool:
        """
        Check if a person is irrelevant for the path between two other people.
        
        This method determines if a person is not on any shortest path between
        person1 and person2, making them irrelevant for the resistance calculation.
        
        Parameters
        ----------
        person : str
            The person to check for relevance.
        person1 : str
            The first person in the path.
        person2 : str
            The second person in the path.
        
        Returns
        -------
        bool
            True if the person is irrelevant (not on any shortest path), False otherwise.
        """
        try:
            path1 = nx.algorithms.shortest_path(self.wordNet,
                                                source = person1, target = person,
                                                weight='weight')
            path2 = nx.algorithms.shortest_path(self.wordNet,
                                                source = person, target = person2,
                                                weight='weight')
        except nx.NetworkXNoPath:
            return True
        intersection_paths = list(set(path1) & set(path2))
        return (len(intersection_paths) != 1)

    def initloop(self, person1: str, person2: str) -> dict[str, float]:
        """
        Initialize voltage values for all nodes in the network.
        
        This method sets initial voltage values for each node based on their
        distance from person1 and person2. Person1 is set to 1.0V, person2 to 0.0V,
        and other nodes are assigned values based on their relative distances.
        
        Parameters
        ----------
        person1 : str
            The person at 1.0V potential.
        person2 : str
            The person at 0.0V potential.
        
        Returns
        -------
        dict
            A dictionary mapping node identifiers to their initial voltage values.
        """
        volDict = {}
        for node in self.wordNet:
            if node == person1:
                volDict[node] = 1.0
                continue
            elif node == person2:
                volDict[node] = 0.0
                continue
            elif self.checkPersonIrrelevant(node, person1, person2):
                volDict[node] = 10.0
                continue
            if self.precalculated_distance:
                distFrom1 = self.distance_matrix[person1, node]
                distFrom2 = self.distance_matrix[node, person2]
            else:
                distFrom1 = float(nx.shortest_path_length(self.wordNet, person1, node, weight='weight'))
                distFrom2 = float(nx.shortest_path_length(self.wordNet, node, person2, weight='weight'))
            volDict[node] = distFrom2 / (distFrom1 + distFrom2)
        return volDict

    def compute_incurrent(self, node: str, volDict: dict[str, float]) -> float:
        """
        Compute the total current flowing into a node.
        
        This method calculates the sum of currents flowing into a node from all
        its predecessors where the predecessor has a higher voltage potential.
        
        Parameters
        ----------
        node : str
            The node for which to compute incoming current.
        volDict : dict
            A dictionary mapping node identifiers to their voltage values.
        
        Returns
        -------
        float
            The total incoming current to the node.
        """
        in_current = 0
        for pred in self.wordNet.predecessors(node):
            if (volDict[pred] > volDict[node]) and (volDict[pred] >= 0.0) and (volDict[pred] <= 1.0):
                potDiff = volDict[pred] - volDict[node]
                resEdge = self.wordNet[pred][node]['weight']
                in_current += potDiff / resEdge
        return in_current

    def compute_outcurrent(self, node: str, volDict: dict[str, float]) -> float:
        """
        Compute the total current flowing out of a node.
        
        This method calculates the sum of currents flowing out of a node to all
        its successors where the node has a higher voltage potential.
        
        Parameters
        ----------
        node : str
            The node for which to compute outgoing current.
        volDict : dict
            A dictionary mapping node identifiers to their voltage values.
        
        Returns
        -------
        float
            The total outgoing current from the node.
        """
        out_current = 0
        for succ in self.wordNet.successors(node):
            if (volDict[node] > volDict[succ]) and (volDict[succ] >= 0.0) and (volDict[succ] <= 1.0):
                potDiff = volDict[node] - volDict[succ]
                resEdge = self.wordNet[node][succ]['weight']
                out_current += potDiff / resEdge
        return out_current

    def average_VR(self, node: str, volDict: dict[str, float]) -> tuple[float, float]:
        """
        Compute the average voltage-to-resistance ratio for a node.
        
        This method calculates the sum of voltage-over-resistance values for all
        connected nodes and the total reciprocal resistance, which are used to
        compute the new voltage potential for the node.
        
        Parameters
        ----------
        node : str
            The node for which to compute the average voltage-to-resistance ratio.
        volDict : dict
            A dictionary mapping node identifiers to their voltage values.
        
        Returns
        -------
        tuple
            A tuple containing:
            - sumVOverR (float): The sum of voltage-over-resistance values.
            - numRecR (float): The total reciprocal resistance.
        """
        sumVOverR = 0.0
        numRecR = 0.0
        for pred in self.wordNet.predecessors(node):
            if (volDict[pred] > volDict[node]) and (volDict[pred] >= 0.0) and (volDict[pred] <= 1.0):
                resEdge = self.wordNet[pred][node]['weight']
                sumVOverR += volDict[pred] / resEdge
                numRecR += 1. / resEdge
        for succ in self.wordNet.successors(node):
            if (volDict[node] > volDict[succ]) and (volDict[succ] >= 0.0) and (volDict[succ] <= 1.0):
                resEdge = self.wordNet[node][succ]['weight']
                sumVOverR += volDict[succ] / resEdge
                numRecR += 1. / resEdge
        return sumVOverR, numRecR

    def getResistance(self, person1: str, person2: str, printVol: bool=False) -> float:
        """
        Compute the resistance distance between two people in the social network.
        
        This method simulates voltage distribution in the network with person1 at
        1.0V and person2 at 0.0V, then calculates the effective resistance between them.
        
        Parameters
        ----------
        person1 : str
            The first person (at 1.0V potential).
        person2 : str
            The second person (at 0.0V potential).
        printVol : bool, optional
            Whether to print voltage values during iteration. Default is False.
        
        Returns
        -------
        float
            The resistance distance between the two people.
        """
        if person1 == person2:
            return 0.0
        if self.precalculated_distance:
            if self.distance_matrix[(person1, person2)] == float('inf'):
                return float('inf')
        else:
            try:
                distTwoWords = nx.shortest_path_length(self.wordNet, person1, person2, weight='weight')
            except nx.exception.NetworkXNoPath:
                return float('inf')

        # initialization
        volDict = self.initloop(person1, person2)
        if printVol:
            print(volDict)
        tempVolDict = {node: volDict[node] for node in self.wordNet}

        # iteration: computing the potential of each node
        converged = False
        step = 0
        while (not converged) and step < self.maxSteps:
            tempConverged = True
            for node in self.wordNet:
                if node == person1:
                    tempVolDict[node] = 1.0
                    continue
                elif node == person2:
                    tempVolDict[node] = 0.0
                    continue
                elif (volDict[node] < 0.0) or (volDict[node] > 1.0):
                    tempVolDict[node] = 10.0
                    continue
                in_current = self.compute_incurrent(node, volDict)
                out_current = self.compute_outcurrent(node, volDict)
                if abs(in_current - out_current) > self.errTol:
                    sumVOverR, numRecR = self.average_VR(node, volDict)
                    tempVolDict[node] = 0.0 if numRecR==0 else sumVOverR / numRecR
                    tempConverged = False
                else:
                    tempConverged = tempConverged and True
            converged = tempConverged
            # value update
            for node in self.wordNet:
                volDict[node] = tempVolDict[node]
            step += 1
            if printVol:
                print(volDict)

        # calculating the resistance
        startCurrent = sum([(1.0-volDict[rootsucc])/self.wordNet[person1][rootsucc]['weight']
                            for rootsucc in self.wordNet.successors(person1) if volDict[rootsucc]<=1.0])
        return (1.0 / startCurrent)
                                
    def drawNetwork(self) -> None:
        """
        Draw the social network using NetworkX.
        
        This method visualizes the social network graph using NetworkX's drawing functions.
        """
        nx.draw(self.wordNet)
        

        

