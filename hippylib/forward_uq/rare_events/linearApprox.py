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

from dolfin                   import Vector, Function
from scipy.stats              import norm
from .bimcOptProblem          import BIMCOptProblem
from ...modeling.variables    import *
from ...modeling.model        import Model
import numpy as np

class LinearApproximator:
    """
    Class that holds linear approximation of an 
    operator, :math:`A(m) ~= A(m0) + J * (m - m0)`
    """

    def __init__(self, m0, opt_problem):
        """
        Constructor

        Inputs:

        - :code:`m0`          : linearization point, a :code:`dolfin.Vector` instance

        - :code:`opt_problem` : :code:`BIMCOptProblem` instance that contains the forward map

        """

        self.m0          = m0
        self.opt_problem = opt_problem
        self.prior       = self.opt_problem.model.prior
        
        self.jac = self.opt_problem.model.generate_vector(PARAMETER)
        self.jac.zero()

        self.b           = None
        self.lin_mean    = None
        self.lin_std_dev = None

    def build(self):
        """
        Build the linear approximation. This involves
        solving the adjoint equation to obtain the 
        jacobian matrix.
        """


        self.opt_problem.getJacobian(self.m0, self.jac)
        self.b = self.opt_problem.getQOI(self.m0) - self.jac.inner(self.m0)

        #Compute R^{-1}J
        R_inv_jac = Vector()
        self.prior.Rsolver.init_vector(R_inv_jac, 0)
        self.prior.Rsolver.solve(R_inv_jac, self.jac)

        self.lin_mean    = self.eval(self.prior.mean)
        self.lin_std_dev = np.sqrt(self.jac.inner(R_inv_jac))

    def eval(self, m):
        """
        Evaluate the linearization at some point :code:`m`

        Inputs:
        
        - :code:`m` : Point at which to evaluate the linearized estimate

        """

        return self.jac.inner(m) + self.b

    def err(self, m):
        """
        Get relative error in using linear approximation at :code:`m`

        Inputs:

        - :code:`m` : Point at which to estimate error

        """

        true_val = getQOI(self.model, m)
        lin_val = self.eval(m)

        return np.fabs((true_val - lin_val) / true_val)

    def getLinearizedProb(self, target_min, target_max):
        """
        Compute the rare event probability using the linearized estimate

        Inputs:

        - :code:`target_min` : lower limit of the target interval

        - :code:`target_max` : upper limit of the target interval
        
        Outputs:
        
        - :code:`prob` : Linearized rare event probability estimate
        """

        d_max = (target_max - self.lin_mean) / self.lin_std_dev
        d_min = (target_min - self.lin_mean) / self.lin_std_dev
        prob = norm.cdf(d_max) - norm.cdf(d_min)

        return prob
