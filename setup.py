
from setuptools import setup
import numpy as np
from Cython.Build import cythonize

dynprog_ext_modules = cythonize(['graphflow/pagerank/cpagerank.pyx'])


def readme():
    with open('README.md') as f:
        return f.read()


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


setup(name='graphflow',
      version="0.4.4",
      description="Algorithms for Graph Flow Analysis",
      long_description="Numerical routines for analyzing data represented by graphs",
      classifiers=[
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
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
                'graphflow.hits'],
      package_data={'graphflow': ['pagerank/*.c', 'pagerank/*.pyx'],
                    'test': ['*.csv']},
      setup_requires=['numpy', 'Cython'],
      install_requires=install_requirements(),
      tests_require=[
          'pandas',
      ],
      include_dirs=[np.get_include()],
      ext_modules=dynprog_ext_modules,
      include_package_data=True,
      test_suite="test",
      zip_safe=False)

