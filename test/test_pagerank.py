
import unittest

import networkx as nx
import graphflow
import graphflow.pagerank.GooglePageRank
from graphflow import PageRankLanguage

nodes = ['Stephen', 'John', 'Mary',
         'Joshua',
         'Abigail', 'Andrew', 'Jacob', 'Melanie',
         'Shirley', 'Zoe', 'Wallace', 'Susan',
         'Urban']
edges = [('Stephen', 'Jacob', 1),
         ('Jacob', 'Stephen', 1),
         ('Stephen', 'Abigail', 1),
         ('Abigail', 'Stephen', 1),
         ('Stephen', 'Andrew', 1),
         ('Andrew', 'Stephen', 1),
         ('Andrew', 'Abigail', 1),
         ('Abigail', 'Andrew', 1),
         ('John', 'Stephen', 1),
         ('Andrew', 'John', 0.4),
         ('John', 'Andrew', 0.6),
         ('Abigail', 'John', 1),
         ('John', 'Abigail', 1),
         ('John', 'Mary', 1),
         ('Mary', 'John', 0.9),
         ('John', 'Joshua', 1),
         ('Joshua', 'John', 1),
         ('John', 'Jacob', 1),
         ('Jacob', 'John', 1),
         ('Abigail', 'Jacob', 1),
         ('Jacob', 'Abigail', 1),
         ('Jacob', 'Andrew', 1),
         ('Andrew', 'Jacob', 1),
         ('Shirley', 'Stephen', 1),
         ('Stephen', 'Shirley', 1),
         ('Melanie', 'Stephen', 1),
         ('Stephen', 'Melanie', 1),
         ('Melanie', 'Shirley', 1),
         ('Shirley', 'Urban', 0.2),
         ('Urban', 'Shirley', 0.21),
         ('Susan', 'Shirley', 1),
         ('Shirley', 'Susan', 1),
         ('Shirley', 'Zoe', 1),
         ('Zoe', 'Shirley', 1),
         ('Shirley', 'Wallace', 1),
         ('Wallace', 'Shirley', 1),
         ('Zoe', 'Wallace', 1)]

pagerank_answer = {'Stephen': 0.08467827,
                   'John': 0.094254784,
                   'Mary': 0.06774095,
                   'Joshua': 0.06774095,
                   'Abigail': 0.075979844,
                   'Andrew': 0.075979844,
                   'Jacob': 0.075979844,
                   'Melanie': 0.06792504,
                   'Shirley': 0.10984976,
                   'Zoe': 0.06867991,
                   'Wallace': 0.07383103,
                   'Susan': 0.06867991,
                   'Urban': 0.06867991}


class test_pagerank(unittest.TestCase):
    def setUp(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges)

        self.googlematrix, self.nodedict = graphflow.pagerank.GooglePageRank.GoogleMatrix(graph, 0.15)

    def testNetwork_fortran(self):
        self.assertRaises(
            ValueError,
            graphflow.pagerank.GooglePageRank.CalculatePageRankFromAdjacencyMatrix,
            adjMatrix=self.googlematrix,
            nodes=self.nodedict,
            language=PageRankLanguage.FORTRAN
        )

    def testNetwork_python(self):
        pagerank = graphflow.pagerank.GooglePageRank.CalculatePageRankFromAdjacencyMatrix(self.googlematrix, self.nodedict, language=PageRankLanguage.PYTHON)

        self.assertEqual(len(pagerank), len(pagerank_answer))
        self.assertEqual(len(set(pagerank.keys()).intersection(set(pagerank_answer.keys()))), len(pagerank))
        for name in pagerank:
            self.assertAlmostEqual(pagerank[name], pagerank_answer[name], places=5)

    def testNetwork_cython(self):
        pagerank = graphflow.pagerank.GooglePageRank.CalculatePageRankFromAdjacencyMatrix(self.googlematrix, self.nodedict, language=PageRankLanguage.CYTHON)

        self.assertEqual(len(pagerank), len(pagerank_answer))
        self.assertEqual(len(set(pagerank.keys()).intersection(set(pagerank_answer.keys()))), len(pagerank))
        for name in pagerank:
            self.assertAlmostEqual(pagerank[name], pagerank_answer[name], places=5)



if __name__ == '__main__':
    unittest.main()
