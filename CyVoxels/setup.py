from distutils.core import setup, Extension
from Cython.Build import cythonize
from os.path import basename, dirname, abspath, join

import numpy


MODULE_NAME = "cyvoxels"


def configure_module(namespace='', top_folder='', compiler_args=None):
    current_dir = basename(dirname(abspath(__file__)))
    sources = ["cyvoxels.pyx"]

    name = "{}.{}".format(namespace, MODULE_NAME) if namespace else MODULE_NAME

    module_sources = [join(top_folder, current_dir, src) for src in sources] if top_folder else sources

    compiler_args = [] if compiler_args is None else compiler_args

    c_module = Extension(
        name,
        sources=module_sources,
        include_dirs=[numpy.get_include()],
        extra_compile_args=compiler_args
    )
    return c_module


if __name__ == '__main__':
    setup(
        name=MODULE_NAME,
        version="0.1",
        description="Provides Voxelization function in C",
        ext_modules=cynthonize(configure_module())
    )
