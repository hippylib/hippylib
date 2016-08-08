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

from variables import STATE, PARAMETER, ADJOINT
from dolfin import Vector, PETScKrylovSolver

class ReducedHessian:
    """
    This class implements matrix free application of the reduced hessian operator.
    The constructor takes the following parameters:
    - model:               the object which contains the description of the problem.
    - innerTol:            the relative tolerance for the solution of the incremental
                           forward and adjoint problems.
    - gauss_newton_approx: a boolean flag that describes whenever the true hessian or
                           the Gauss Newton approximation of the Hessian should be
                           applied.
    - misfit_only:         a boolean flag that describes whenever the full hessian or
                           only the misfit component of the hessian is used.
    
    Type help(modelTemplate) for more information on which methods model should implement.
    """
    def __init__(self, model, innerTol, gauss_newton_approx=False, misfit_only=False):
        """
        Construct the reduced Hessian Operator:
        """
        self.model = model
        self.tol = innerTol
        self.gauss_newton_approx = gauss_newton_approx
        self.misfit_only=misfit_only
        self.ncalls = 0
        
        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(ADJOINT)
        self.rhs_adj2 = model.generate_vector(ADJOINT)
        self.uhat    = model.generate_vector(STATE)
        self.phat    = model.generate_vector(ADJOINT)
        self.yhelp = model.generate_vector(PARAMETER)
    
    def init_vector(self, x, dim):
        """
        Reshape the Vector x so that it is compatible with the reduced Hessian
        operator.
        Parameters:
        - x: the vector to reshape
        - dim: if 0 then x will be reshaped to be compatible with the range of
               the reduced Hessian
               if 1 then x will be reshaped to be compatible with the domain of
               the reduced Hessian
               
        Note: Since the reduced Hessian is a self adjoint operator, the range and
              the domain is the same. Either way, we choosed to add the parameter
              dim for consistency with the interface of Matrix in dolfin.
        """
        self.model.init_parameter(x)
        
    def mult(self,x,y):
        """
        Apply the reduced Hessian (or the Gauss Newton approximation) to the vector x
        Return the result in y.
        """
        if self.gauss_newton_approx:
            self.GNHessian(x,y)
        else:
            self.TrueHessian(x,y)
        
        self.ncalls += 1
    
    def inner(self,x,y):
        """
        Perform the inner product between x and y in the norm induced by the reduced
        Hessian H.
        (x, y)_H = x' H y
        """
        Ay = self.model.generate_vector(PARAMETER)
        Ay.zero()
        self.mult(y,Ay)
        return x.inner(Ay)
            
    def GNHessian(self,x,y):
        """
        Apply the Gauss Newton approximation of the reduced Hessian to the vector x
        Return the result in y.        
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd, self.tol)
        self.model.applyWuu(self.uhat, self.rhs_adj, True)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj, self.tol)
        self.model.applyCt(self.phat, y)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.axpy(1., self.yhelp)

        
    def TrueHessian(self, x, y):
        """
        Apply the the reduced Hessian to the vector x.
        Return the result in y.        
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd, self.tol)
        self.model.applyWuu(self.uhat, self.rhs_adj, False)
        self.model.applyWua(x, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj, self.tol)
        self.model.applyRaa(x, y)
        self.model.applyCt(self.phat, self.yhelp)
        y.axpy(1., self.yhelp)
        self.model.applyWau(self.uhat, self.yhelp)
        y.axpy(-1., self.yhelp)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.axpy(1., self.yhelp)