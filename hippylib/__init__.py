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

"""
hIPPYlib implements state-of-the-art scalable algorithms for PDE-based
deterministic and Bayesian inverse problems. It builds on 
FEniCS (http://fenicsproject.org/) (a parallel finite element element library) 
for the discretization of the PDE and on PETSc (http://www.mcs.anl.gov/petsc/)
for scalable and efficient linear algebra operations and solvers.

For building instructions, see the file INSTALL. Copyright information
and licensing restrictions can be found in the file COPYRIGHT.

The best starting point for new users interested in hIPPYlib's features are the
interactive tutorials in the notebooks folder.

Conceptually, hIPPYlib can be viewed as a toolbox that provides
the building blocks for experimenting new ideas and developing scalable
algorithms for PDE-based deterministic and Bayesian inverse problems.
"""
from __future__ import absolute_import, division, print_function


# version
from .version import version_info, __version__

# utils
from .utils import *

# hIPPYlib model
from .modeling import *

# hIPPYlib algorithms
from .algorithms import *

#MCMC:
from .mcmc import *
