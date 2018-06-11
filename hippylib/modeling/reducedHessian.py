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

from .variables import STATE, PARAMETER, ADJOINT

class ReducedHessian:
    """
    This class implements matrix free application of the reduced Hessian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    - :code:`innerTol`:            the relative tolerance for the solution of the incremental forward and adjoint problems.
    - :code:`misfit_only`:         a boolean flag that describes whenever the full Hessian or only the misfit component of the Hessian is used.
    
    Type :code:`help(modelTemplate)` for more information on which methods model should implement.
    """
    def __init__(self, model, innerTol, misfit_only=False):
        """
        Construct the reduced Hessian Operator
        """
        self.model = model
        self.tol = innerTol
        self.gauss_newton_approx = self.model.gauss_newton_approx 
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
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

        - :code:`x`: the vector to reshape.
        - :code:`dim`: if 0 then :code:`x` will be reshaped to be compatible with the range of the reduced Hessian, if 1 then :code:`x` will be reshaped to be compatible with the domain of the reduced Hessian.
               
        .. note:: Since the reduced Hessian is a self adjoint operator, the range and the domain is the same. Either way, we choosed to add the parameter :code:`dim` for consistency with the interface of :code:`Matrix` in dolfin.
        """
        self.model.init_parameter(x)
        
    def mult(self,x,y):
        """
        Apply the reduced Hessian (or the Gauss-Newton approximation) to the vector :code:`x`. Return the result in :code:`y`.
        """
        if self.gauss_newton_approx:
            self.GNHessian(x,y)
        else:
            self.TrueHessian(x,y)
        
        self.ncalls += 1
    
    def inner(self,x,y):
        """
        Perform the inner product between :code:`x` and :code:`y` in the norm induced by the reduced
        Hessian :math:`H,\\,(x, y)_H = x' H y`.
        """
        Ay = self.model.generate_vector(PARAMETER)
        Ay.zero()
        self.mult(y,Ay)
        return x.inner(Ay)
            
    def GNHessian(self,x,y):
        """
        Apply the Gauss-Newton approximation of the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.        
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd, self.tol)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj, self.tol)
        self.model.applyCt(self.phat, y)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.axpy(1., self.yhelp)

        
    def TrueHessian(self, x, y):
        """
        Apply the the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.        
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd, self.tol)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWum(x, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj, self.tol)
        self.model.applyWmm(x, y)
        self.model.applyCt(self.phat, self.yhelp)
        y.axpy(1., self.yhelp)
        self.model.applyWmu(self.uhat, self.yhelp)
        y.axpy(-1., self.yhelp)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.axpy(1., self.yhelp)
            
            
class FDHessian:
    """
    This class implements matrix free application of the reduced Hessian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    - :code:`m0`:                  the value of the parameter at which the Hessian needs to be evaluated.
    - :code:`h`:                   the mesh size for FD.
    - :code:`innerTol`:            the relative tolerance for the solution of the forward and adjoint problems.
    - :code:`misfit_only`:         a boolean flag that describes whenever the full Hessian or only the misfit component of the Hessian is used.
    
    Type :code:`help(Template)` for more information on which methods model should implement.
    """
    def __init__(self, model, m0, h, innerTol,  misfit_only=False):
        """
        Construct the reduced Hessian Operator
        """
        self.model = model
        self.m0 = m0.copy()
        self.h = h
        self.tol = innerTol
        self.misfit_only=misfit_only
        self.ncalls = 0
        
        self.state_plus  = model.generate_vector(STATE)
        self.adj_plus    = model.generate_vector(ADJOINT)
        self.state_minus = model.generate_vector(STATE)
        self.adj_minus   = model.generate_vector(ADJOINT)
        self.g_plus      = model.generate_vector(PARAMETER)
        self.g_minus     = model.generate_vector(PARAMETER)
        self.yhelp       = model.generate_vector(PARAMETER)
    
    def init_vector(self, x, dim):
        """
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

        - :code:`x`: the vector to reshape
        - :code:`dim`: if 0 then :code:`x` will be reshaped to be compatible with the range of the reduced Hessian, if 1 then :code:`x` will be reshaped to be compatible with the domain of the reduced Hessian
               
        .. note:: Since the reduced Hessian is a self adjoint operator, the range and the domain is the same. Either way, we choosed to add the parameter :code:`dim` for consistency with the interface of :code:`Matrix` in dolfin.
        """
        self.model.init_parameter(x)
        
    def mult(self,x,y):
        """
        Apply the reduced Hessian (or the Gauss-Newton approximation) to the vector :code:`x`.
        Return the result in :code:`y`.
        """
        h = self.h
        
        m_plus = self.m0.copy()
        m_plus.axpy(h, x)
        self.model.solveFwd(self.state_plus, [self.state_plus, m_plus, self.adj_plus], self.tol)
        self.model.solveAdj(self.adj_plus, [self.state_plus, m_plus, self.adj_plus], self.tol)
        self.model.evalGradientParameter([self.state_plus, m_plus, self.adj_plus], self.g_plus, misfit_only = True)
        
        m_minus = self.m0.copy()
        m_minus.axpy(-h, x)
        self.model.solveFwd(self.state_minus, [self.state_minus, m_minus, self.adj_minus], self.tol)
        self.model.solveAdj(self.adj_minus, [self.state_minus, m_minus, self.adj_minus], self.tol)
        self.model.evalGradientParameter([self.state_minus, m_minus, self.adj_minus], self.g_minus, misfit_only = True)
        
        y.zero()
        y.axpy(1., self.g_plus)
        y.axpy(-1., self.g_minus)
        y*=(.5/h)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.axpy(1., self.yhelp)

        
        self.ncalls += 1
    
    def inner(self,x,y):
        """
        Perform the inner product between :code:`x` and :code:`y` in the norm induced by the reduced Hessian :math:`H,\\, (x, y)_H = x' H y`.
        """
        Ay = self.model.generate_vector(PARAMETER)
        Ay.zero()
        self.mult(y,Ay)
        return x.inner(Ay)
