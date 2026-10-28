
import numpy as np
import networkx as nx
from graphflow.hits import calculate_hits
import pytest


def test_fork():
    forkgr = nx.DiGraph()
    forkgr.add_nodes_from(['a', 'b', 'c'])
    forkgr.add_edges_from([('a', 'b'), ('c', 'b')])

    hubdict, authdict = calculate_hits(forkgr)

    assert hubdict['a'] == pytest.approx(np.sqrt(0.5))
    assert hubdict['b'] == pytest.approx(0.0)
    assert hubdict['c'] == pytest.approx(np.sqrt(0.5))
    assert authdict['a'] == pytest.approx(0.0)
    assert authdict['b'] == pytest.approx(1.0)
    assert authdict['c'] == pytest.approx(0.0)
