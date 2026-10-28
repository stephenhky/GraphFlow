
import numpy as np
import numba as nb


@nb.njit(nb.float64[nb.float64[:, :], nb.float64[:, :]])
def L1norm(r1, r2):
    return np.sum(np.abs(r1 - r2))


from . import hits
from . import pagerank
from . import simvoltage
