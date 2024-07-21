
import unittest

import numpy as np
import networkx as nx
import graphflow
import graphflow.hits


class testHITS(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFork(self):
        forkgr = nx.DiGraph()
        forkgr.add_nodes_from(['a', 'b', 'c'])
        forkgr.add_edges_from([('a', 'b'), ('c', 'b')])

        hubdict, authdict = graphflow.hits.CalculateHITS(forkgr)

        self.assertAlmostEqual(hubdict['a'], np.sqrt(0.5))
        self.assertAlmostEqual(hubdict['b'], 0.0)
        self.assertAlmostEqual(hubdict['c'], np.sqrt(0.5))
        self.assertAlmostEqual(authdict['a'], 0.0)
        self.assertAlmostEqual(authdict['b'], 1.0)
        self.assertAlmostEqual(authdict['c'], 0.0)


if __name__ == '__main__':
    unittest.main()
