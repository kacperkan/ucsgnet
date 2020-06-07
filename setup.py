# Build with `python setup.py build_ext --inplace`

from distutils.core import setup

import numpy as np
from Cython.Build import cythonize

setup(
    name="ucsg",
    ext_modules=cythonize(
        "ucsgnet/mesh_utils.pyx", compiler_directives={"language_level": "3"}
    ),
    include_dirs=[np.get_include()],
)
