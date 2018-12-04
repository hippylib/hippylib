# Copyright (c) 2016-2018, The University of Texas at Austin
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.


from setuptools import setup, find_packages
from os import path
from io import open
import re


try:
    from dolfin import __version__ as dolfin_version
    from dolfin import has_linear_algebra_backend, has_slepc
except ImportError:
    raise
else:
    fenics_version_major = int(dolfin_version[0:4])
    if (fenics_version_major < 2016 or fenics_version_major >= 2018):
        raise Exception(
            'hIPPYlib requires FEniCS versions 2016.x.x or 2017.x.x')
    if not has_linear_algebra_backend("PETSc"):
        raise Exception(
            'hIPPYlib requires FEniCS to be installed with PETSc support')
    if not has_slepc():
        raise Exception(
            'hIPPYlib requires FEniCS to be installed with SLEPc support')
    

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

version = {}
with open(path.join(here, 'hippylib/version.py')) as f:
    exec(f.read(), version)

VERSION = version['__version__']

REQUIREMENTS = [
    'mpi4py',
    'numpy',
    'matplotlib',
    'scipy'
]

EXTRAS = {
    'Notebook':  ["jupyter"],
}

KEYWORDS = """
    Infinite-dimensional inverse problems, 
    adjoint-based methods, numerical optimization, 
    low-rank approximation, Bayesian inference, 
    uncertainty quantification, sampling"""

setup(
    name='hippylib',
    version=VERSION,
    description='an Extensible Software Framework for Large-scale Deterministic and Bayesian Inverse Problems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://hippylib.github.io/',
    author='Umberto Villa, Noemi Petra, Omar Ghattas',
    author_email='uvilla@wustl.edu',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=KEYWORDS,
    packages=find_packages(exclude=['applications', 'doc', 'test']),
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS,
    python_requires=">=2.6,!=3.0,!=3.1,!=3.2,!=3.3,!=3.4,!=3.5,<4",
    include_package_data=True,
    package_data={
        'hippylib': [
            'utils/cpp_rand/*.cpp', 'utils/cpp_rand/*.h',
            'modeling/cpp_AssemblePointwiseObservation/*.cpp', 'modeling/cpp_AssemblePointwiseObservation/*.h',
            'algorithms/cpp_multivector/*.cpp', 'algorithms/cpp_multivector/*.h',
            'algorithms/cpp_linalg/*.cpp', 'algorithms/cpp_linalg/*.h',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/hippylib/hippylib/issues',
        'Source': 'https://github.com/hippylib/hippylib',
    },
)
