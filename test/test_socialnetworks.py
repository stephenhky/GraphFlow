
from pathlib import Path

import numpy as np
import pandas as pd
from graphflow.simvoltage import SocialNetworkSimVoltage, GraphResistanceDistance
import pytest


def test_socialnetwork():
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

    wn1 = SocialNetworkSimVoltage(nodes=nodes, edges=edges, precalculated_distance=True)

    this_dir = Path(__file__).absolute().parent
    testresults = pd.read_csv(
        this_dir / 'socialnetworkranks.csv',
        header=None,
        names=['name1', 'name2', 'resistance'],
        dtype={'name1': str, 'name2': str, 'resistance': np.float64}
    )

    for _, row in testresults.iterrows():
        name1 = row['name1']
        name2 = row['name2']
        resistance = row['resistance']
        assert wn1.get_resistance(name1, name2) == pytest.approx(resistance, abs=1e-3)
        print(f'{name1}\t{name2} : {resistance:.4f} passed.')


def test_resistdist():
    obj = GraphResistanceDistance()
    assert obj.get_resistance('Stephen', 'Sinnie') == pytest.approx(0.6666666666666667, abs=1e-4)
    assert obj.get_resistance('Elaine', 'Sinnie') == pytest.approx(0.6666666666666667, abs=1e-4)
    assert obj.get_resistance('Elaine', 'Stephen') ==pytest.approx(0.6666666666666667, abs=4)
