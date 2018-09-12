
import numpy as np

def L1norm(r1, r2):
    return np.sum(abs(r1 - r2))

from . import simvoltage
from . import pagerank
from . import hits
