
from enum import Enum

import numpy as np


def L1norm(r1, r2):
    return np.sum(np.abs(r1 - r2))


class PageRankLanguage(Enum):
    PYTHON = 0
    CYTHON = 1
    FORTRAN = 2


from . import hits
from . import pagerank
from . import simvoltage
