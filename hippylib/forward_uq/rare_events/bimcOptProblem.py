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

import numpy as np

from dolfin                              import Vector
from ...modeling.misfit                  import QOIMisfit
from ...modeling.variables               import *
from ...modeling.model                   import Model
from ...modeling.posterior               import GaussianLRPosterior, LowRankHessian
from ...modeling.reducedHessian          import ReducedHessian
from ...algorithms.NewtonCG              import ReducedSpaceNewtonCG
from ...algorithms.bfgs                  import BFGS
from ...algorithms.multivector           import MultiVector
from ...algorithms.randomizedEigensolver import doublePassG
from ...utils.random                     import parRandom


class NotConvergedException(Exception):
    pass


class BIMCOptProblem(object):
    """

    Abstract base class for a BIMC optimization problem. A concrete instance of
    :code:`BIMCOptProblem` must expose the following:

    Attributes:

    - :code:`model`  : A :code:`hippylib.model`
    - :code:`solver` : A :code:`hippylib.solver`

    """
    
    def getQOIFromState(self, x):
        
        """
        Get QoI from solution vector x

        Inputs: 
        
        - :code:`x` : a :code:`hippylib` vector
        
        Outputs:
        
        - QOI

        """
        
        return self.model.misfit.qoi.eval(x)
       
    def getStateFromParam(self, m):
        """
        Returns state vector from m 
        vector by solving the forward equation

        Inputs:

        - :code:`m`  : parameter, a :code:`dolfin.Vector` instance

        Outputs:

        - State corresponding to the parameter, and :code:`hippylib` solution  vector

        """
        [u, b, p] = self.model.generate_vector()
        self.model.solveFwd(u, [u, m, p])
        return (u, [u, m, p])
    
    
    def getQOI(self, m):
        """
        Returns QoI from parameter m

        Inputs:

        - :code:`m` : parameter, a :code:`dolfin.Vector` instance

        Outputs:

        - QOI

        """
    
        u, vec = self.getStateFromParam(m) 
        return self.getQOIFromState(vec)
     
    def getJacobian(self, m, jac):
        """
        Get jacobian of p2o map at :code:`m`

        Inputs:

        - :code:`m` : parameter at which to compute the jacobian, a :code:`dolfin.Vector` instance

        - :code:`jac` : :code:`dolfin.Vector` instance that will store the Jacobian

        """
    
        u, vec = self.getStateFromParam(m)
        self.model.problem.setLinearizationPoint(vec, gauss_newton_approx=True)
        rhs = self.model.generate_vector(STATE)
       
        grad = self.model.generate_vector(STATE)
        self.model.misfit.qoi.grad_state(vec, grad)
        rhs.axpy(-1.0, grad)
    
        #init p_hat
        p_hat = self.model.generate_vector(ADJOINT)
        self.model.problem.solveIncremental(p_hat, rhs, is_adj=True, mytol=1e-9)
        self.model.applyCt(p_hat, jac)
   
    def _setMisfit(self, target):
        """
        Set misfit observation to target

        Inputs:

        - :code:`target`: The pseudo data point

        """
        self.model.misfit.d = target
      
            
    def getMAP(self, target, m0, bounds=None):
        """
        Get MAP point corresponding to observation target

        Inputs: 

        - :code:`target` : Pseudo data point

        - :code:`m0`     : Initial guess for the optimizer

        - :code:`bounds` : If solving a bound constrained problem. \
                Solver must be :code:`hippylib.algorithms.BFGS`    

        """
        
        self._setMisfit(target)
        
        #map_sol = self.model.generate_vector()
        if isinstance(self.solver, BFGS) and bounds is not None:
            map_sol = self.solver.solve([None, m0, None], self.model.prior.Rsolver, bounds)
        elif isinstance(self.solver, ReducedSpaceNewtonCG):
            map_sol = self.solver.solve([None, m0, None])
        
        if not self.solver.converged:
            raise NotConvergedException("Failed to obtain MAP\nExit reason %d" \
                                        % self.solver.reason)
    
        return map_sol
    
    def getHessLR(self, x):
        """
        Get a low rank approximation of the Gauss-Newton Hessian
        
        Inputs:

        - :code:`x` : :code:`hippylib` solution vector

        Outputs:

        - :code:`eig_val` : Leading eigenvalues of the prior preconditioned misfit hessian
        - :code:`eig_vec` : Leading eigenvectors of the prior preconditioned misfit hessian
        """
    
        self.model.setPointForHessianEvaluations(x, gauss_newton_approx=True)

        hess = ReducedHessian(self.model, 
                              self.solver.parameters["inner_rel_tolerance"], 
                              misfit_only=True)
    

        k = 1  #desired rank
        p = 10 #oversampling factor
    
        Omega = MultiVector(x[PARAMETER], k+p)
        parRandom.normal(1.0, Omega)
    
        (eig_val, eig_vec) = doublePassG(hess, self.model.prior.R, \
                                       self.model.prior.Rsolver, \
                                       Omega, k, \
                                       s=1, check=False)
    
        return eig_val, eig_vec
    
    
    
