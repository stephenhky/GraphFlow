
import unittest

import numpy as np
import networkx as nx
import graphflow


class testHITS(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFork(self):
        forkgr = nx.DiGraph()
        forkgr.add_nodes_from(['a', 'b', 'c'])
        forkgr.add_edges_from([('a', 'b'), ('c', 'b')])

        A = nx.adj_matrix(forkgr).toarray()
        nodesdict = {node: idx for idx, node in enumerate(forkgr.nodes)}

        i, p = graphflow.hits.hits(A, nodesdict)

        self.assertAlmostEquals(i[nodesdict['a']], np.sqrt(0.5))
        self.assertAlmostEquals(i[nodesdict['b']], 0.0)
        self.assertAlmostEquals(i[nodesdict['c']], np.sqrt(0.5))
        self.assertAlmostEquals(p[nodesdict['a']], 0.0)
        self.assertAlmostEquals(p[nodesdict['b']], 1.0)
        self.assertAlmostEquals(p[nodesdict['c']], 0.0)


if __name__ == '__main__':
    unittest.main()