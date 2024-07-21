
import unittest

import numpy as np
from graphflow.simvoltage import SocialNetworkSimVoltage


class test_SocialNetwork(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simplecircuits(self):
        circuit = SocialNetworkSimVoltage(nodes=['a', 'b'],
                                          edges=[('a', 'b', 13.2)])
        self.assertAlmostEqual(circuit.getResistance('a', 'b'), 13.2)
        self.assertAlmostEqual(circuit.getResistance('b', 'a'), np.inf)

    def test_parallelcircuits(self):
        circuit = SocialNetworkSimVoltage(nodes=['a', 'b', 'c'],
                                          edges=[('a', 'b', 10.0),
                                                 ('b', 'c', 10.0),
                                                 ('a', 'c', 20.0)])
        # self.assertAlmostEquals(circuit.getResistance('a', 'b'), 10.0)
        self.assertAlmostEqual(circuit.getResistance('b', 'c'), 10.0)
        self.assertAlmostEqual(circuit.getResistance('a', 'c'), 10.0)
        self.assertAlmostEqual(circuit.getResistance('c', 'a'), np.inf)

if __name__ == '__main__':
    unittest.main()