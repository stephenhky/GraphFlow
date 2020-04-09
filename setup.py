
# must import thisfirst. Ref: # must import this. Ref: https://stackoverflow.com/questions/7932028/setup-py-for-packages-that-depend-on-both-cython-and-f2py?rq=1
from setuptools import setup

import numpy as np
from numpy.distutils.core import setup, Extension

try:
    from Cython.Build import cythonize
    dynprog_ext_modules = cythonize(['graphflow/pagerank/cpagerank.pyx'])
except ImportError:
    dynprog_ext_modules = [Extension('graphflow.pagerank.cpagerank', ['graphflow/pagerank/cpagerank.c'])]



def readme():
    with open('README.md') as f:
        return f.read()


setup(name='graphflow',
      version="0.3.0",
      description="Algorithms for Graph Flow Analysis",
      long_description="Numerical routines for analyzing data represented by graphs",
      classifiers=[
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Fortran",
          "Programming Language :: Cython",
          "Programming Language :: C",
          "License :: OSI Approved :: MIT License",
      ],
      keywords="Algorithms for Graph Flow Analysis",
      url="https://github.com/stephenhky/GraphFlow",
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['graphflow',
                'graphflow.pagerank',
                'graphflow.simvoltage',
                'graphflow.hits',],
      package_data={'graphflow': ['pagerank/*.f90', 'pagerank/*.pyf',
                                  'pagerank/*.c', 'pagerank/*.pyx'],
                    'test': ['*.csv']},
      setup_requires=['numpy', 'Cython'],
      install_requires=[
          'Cython', 'numpy', 'scipy', 'networkx>=2.0',
      ],
      tests_require=[
          'unittest2', 'pandas',
      ],
      include_dirs=[np.get_include()],
      ext_modules = [Extension( 'f90pagerank', sources=['graphflow/pagerank/f90pagerank.f90',
                                                        'graphflow/pagerank/f90pagerank.pyf']),
                     ] + dynprog_ext_modules,
      include_package_data=True,
      test_suite="test",
      zip_safe=False)

