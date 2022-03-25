import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Fringe',
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("fringe_analysis.pyx")
)
