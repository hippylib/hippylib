# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

"""
hIPPYlib implements state-of-the-art scalable algorithms for PDE-based
deterministic and Bayesian inverse problems. It builds on FEniCS (a 
parallel finite element element library) [http://fenicsproject.org/]
for the discretization of the PDE and on PETSc [http://www.mcs.anl.gov/petsc/]
for scalable and efficient linear algebra operations and solvers.

For building instructions, see the file INSTALL. Copyright information
and licensing restrictions can be found in the file COPYRIGHT.

The best starting point for new users interested in hIPPYlib's features are the
interactive tutorials in the notebooks folder.

Conceptually, hIPPYlib can be viewed as a toolbox that provides
the building blocks for experimenting new ideas and developing scalable
algorithms for PDE-based deterministic and Bayesian inverse problems.
"""

# utils
from expression import code_AnisTensor2D, code_Mollifier
from linalg import MatMatMult, MatPtAP, Transpose, to_dense, trace, get_diagonal, estimate_diagonal_inv2, randn_perturb, amg_method, Solver2Operator, vector2Function
from pointwiseObservation import assemblePointwiseObservation, exportPointwiseObservation
from timeDependentVector import TimeDependentVector


# hIPPYlib model
from variables import *
from PDEProblem import PDEProblem, PDEVariationalProblem
from prior import _Prior, LaplacianPrior, BiLaplacianPrior, MollifiedBiLaplacianPrior
from misfit import Misfit, ContinuousStateObservation, PointwiseStateObservation
from model import Model
from modelTemplate import ModelTemplate, modelVerify 

# hIPPYlib algorithms
from cgsolverSteihaug import CGSolverSteihaug
from NewtonCG import ReducedSpaceNewtonCG
from randomizedEigensolver import singlePass, doublePass, singlePassG, doublePassG
from lowRankOperator import LowRankOperator
from traceEstimator import TraceEstimator
from cgsampler import CGSampler


# hIPPYlib outputs
from reducedHessian import ReducedHessian
from posterior import GaussianLRPosterior, LowRankHessian