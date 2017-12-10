from os import getcwd
from distutils.core import setup
from Cython.Build import cythonize

import CVoxels.setup as CVoxelsSetup
import CyVoxels.setup as CyVoxelsSetup


ext = [
    CVoxelsSetup.configure_module(top_folder=getcwd()),
]
ext.extend(cythonize(CyVoxelsSetup.configure_module(top_folder=getcwd())))

setup(name='PythonExtensions',
      version='1.0',
      ext_modules = ext
      )