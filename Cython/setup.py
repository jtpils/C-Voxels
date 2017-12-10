from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy


extensions = [
    Extension(
        "cyvoxels",
        ["cyvoxels.pyx"],
        include_dirs = [numpy.get_include()]
    )
]

setup(
    ext_modules = cythonize(extensions)
)