
from setuptools import setup
import numpy as np
from Cython.Build import cythonize


dynprog_ext_modules = cythonize(['graphflow/pagerank/cpagerank.pyx'])


setup(
      include_dirs=[np.get_include()],
      ext_modules=dynprog_ext_modules
)
