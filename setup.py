from distutils.core import setup, Extension
import numpy

module = Extension("cvoxels", sources=["c_voxels.c"],
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=['-std=c99'],)

setup(name="PLAnalisys",
      version="0.1",
      description="Provides functions in C",
      ext_modules=[module])
