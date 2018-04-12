
# must import thisfirst. Ref: # must import this. Ref: https://stackoverflow.com/questions/7932028/setup-py-for-packages-that-depend-on-both-cython-and-f2py?rq=1
from setuptools import setup

from numpy.distutils.core import setup, Extension

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='graphflow',
      version="0.1.0",
      description="Algorithms for Graph Flow Analysis",
      long_description="Numerical routines for analyzing data represented by graphs",
      classifiers=[
          "Topic :: Scientific/Engineering :: Mathematics",
          "Programming Language :: Python :: 2.7",
          "License :: OSI Approved :: MIT License",
      ],
      keywords="Algorithms for Graph Flow Analysis",
      url="https://github.com/stephenhky/GraphFlow",
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['graphflow',
                'graphflow.pagerank',
                'graphflow.simvoltage',],
      package_data={'graphflow': ['pagerank/*.f90', 'pagerank/*.pyf'],
                    'test': ['*.csv']},
      setup_requires=['numpy',],
      install_requires=[
          'numpy', 'scipy', 'networkx>=2.0',
      ],
      tests_require=[
          'unittest2', 'pandas',
      ],
      ext_modules = [Extension( 'f90pagerank', sources=['graphflow/pagerank/f90pagerank.f90',
                                                        'graphflow/pagerank/f90pagerank.pyf']),
                     ],
      include_package_data=True,
      test_suite="test",
      zip_safe=False)

