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

from __future__ import absolute_import, division, print_function

import dolfin as dl
import numpy as np
import os
from tempfile import NamedTemporaryFile


import sys
sys.path.append('../../../')
from hippylib import *

class _TestR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the TestPrior class
    """
    def __init__(self, A):
        self.A = A

        self.help1 = dl.Vector()
        self.A.init_vector(self.help1, 0)

    def init_vector(self,x, dim):
        self.A.init_vector(x,1)

    def mpi_comm(self):
        return self.A.mpi_comm()

    def inner(self,x,y):
        Rx = dl.Vector()
        self.init_vector(Rx,0)
        self.mult(x, Rx)
        return Rx.inner(y)

    def mult(self,x,y):
        self.A.mult(x, self.help1)
        self.A.mult(self.help1, y)




class TestPrior(prior._Prior):

    """
    Prior for independent, normally distributed real-valued random variables
    Since the random variables are independent, the covariance matrix is diagonal.
    """

    def __init__(self, Vh, variance=[1.0], mean=None):
        """
        Constructor.
        Inputs:
        -- Vh       : Finite element space for the parameter
        -- variance : Variance of each individual random variable
                      If variance is a list with a single element, the variance
                      in all variables is assumed to be equal. If distinct
                      variances are required, len(variance) must
                      be equal to the number of random variables (the dimension of the
                      function space).
        -- mean     : Mean of random variables
        """

        self.Vh        = Vh
        self.variance  = variance
           

        trial = dl.TrialFunction(Vh)
        test  = dl.TestFunction(Vh)

        fspace_dim = Vh.dim()

        if len(variance) == 1:
            var_list = [variance[0] for i in range(fspace_dim)]
        else:
            assert len(variance) == fspace_dim
            var_list = variance
    
        self.prec_list = [1.0 / var_list[i] for i in range(fspace_dim)]

        varfM = dl.inner(trial, test) * dl.dx

        varfR = sum([self.prec_list[i] * trial[i] * test[i] for i in range(fspace_dim)])
        varfRinv = sum([var_list[i] * trial[i] * test[i] for i in range(fspace_dim)])

        varfRsqrt = sum([np.sqrt(self.prec_list[i]) * trial[i] * test[i] for i in range(fspace_dim)])
        varfRsqrtinv = sum([np.sqrt(var_list[i]) * trial[i] * test[i] for i in range(fspace_dim)])

        #mass matrix
        self.M = dl.assemble(varfM)
        self.Msolver = linalg.Operator2Solver(dl.assemble(varfM))

        self.R = _TestR(dl.assemble(varfR * dl.dx))

        self.RSolverOp = dl.assemble(varfRinv * dl.dx)
        self.Rsolver   = linalg.Operator2Solver(self.RSolverOp)

        self.sqrtR    = dl.assemble(varfRsqrt * dl.dx)
        self.sqrtRinv = dl.assemble(varfRsqrtinv * dl.dx)

        if mean:
            self.mean  = mean
        else:
            tmp = dl.Vector()
            self.M.init_vector(tmp, 0)
            self.mean = tmp
 
    def init_vector(self, x, dim):
        """
        Initialize a vector x to be compatible with the range/domain of R.
        If dim == "noise" initialize x to be compatible with the size of
        white noise used for sampling.
        """

        if dim == "noise":
            self.sqrtR.init_vector(x, 1)
        else:
            self.R.init_vector(x, dim)

    def sample(self, noise, s, add_mean=True):
        """
        Given a noise ~ N(0, I) compute a sample s from the prior.
        If add_mean=True add the prior mean value to s.
        """

        self.sqrtR.mult(noise, s)

        if add_mean:
            s.axpy(1.0, self.mean)




