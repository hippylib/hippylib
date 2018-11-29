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


try:
    from dolfin import __version__ as dolfin_version
except ImportError:
    raise
else:
    if (int(dolfin_version[0:4]) < 2016):
        raise Exception(
            "hippylib requires FEniCS installation not older than 2016.1.0")

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

REQUIREMENTS = [
    "mpi4py",
    "numpy",
    "matplotlib",
    "scipy",
    "sympy==1.1.1",
    "jupyter"
]

setup(
    name='hippylib',
    version='1.7.0.dev',
    description='Baeysian inversion toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://hippylib.github.io/',
    author='Umberto Villa, Noemi Petra, Omar Ghattas',
    author_email='hippylib-dev@googlegroups.com',
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
    keywords='optimization HPC Bayesian inverse',
    packages=find_packages(exclude=['applications', 'doc', 'test']),
    install_requires=REQUIREMENTS,
    python_requires='>=2.6, >=3.6, <4',
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
