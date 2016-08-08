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

import dolfin as dl
import math
from variables import STATE, PARAMETER, ADJOINT

class Model:
    """
    This class contains the full description of the inverse problem.
    As inputs it takes a PDEProblem object, a Prior object, and a Misfit object.
    
    In the following we will denote with
    - u the state variable
    - a the parameter variable
    - p the adjoint variable
        
    """
    
    def __init__(self, problem, prior,misfit):
        """
        Create a model given:
        - problem: the description of the forward/adjoint problem and all the sensitivities
        - prior: the prior component of the cost functional
        - misfit: the misfit componenent of the cost functional
        """
        self.problem = problem
        self.prior = prior
        self.misfit = misfit
                
    def generate_vector(self, component = "ALL"):
        """
        By default, return the list [u,a,p] where:
        - u is any object that describes the state variable
        - a is a Vector object that describes the parameter variable.
          (Need to support linear algebra operations)
        - p is any object that describes the adjoint variable
        
        If component = STATE return only u
           component = PARAMETER return only a
           component = ADJOINT return only p
        """ 
        if component == "ALL":
            a = dl.Vector()
            self.prior.init_vector(a,0)
            x = [self.problem.generate_state(), a, self.problem.generate_state()]
        elif component == STATE:
            x = self.problem.generate_state()
        elif component == PARAMETER:
            x = dl.Vector()
            self.prior.init_vector(x,0)
        elif component == ADJOINT:
            x = self.problem.generate_state()
            
        return x
    
    def init_parameter(self, a):
        """
        Reshape a so that it is compatible with the parameter variable
        """
        self.prior.init_vector(a,0)
            
    def cost(self, x):
        """
        Given the list x = [u,a,p] which describes the state, parameter, and
        adjoint variable compute the cost functional as the sum of 
        the misfit functional and the regularization functional.
        
        Return the list [cost functional, regularization functional, misfit functional]
        
        Note: p is not needed to compute the cost functional
        """
        misfit_cost = self.misfit.cost(x)
        reg_cost = self.prior.cost(x[PARAMETER])
        return [misfit_cost+reg_cost, reg_cost, misfit_cost]
    
    def solveFwd(self, out, x, tol=1e-9):
        """
        Solve the (possibly non-linear) forward problem.
        Parameters:
        - out: is the solution of the forward problem (i.e. the state) (Output parameters)
        - x = [u,a,p] provides
              1) the parameter variable a for the solution of the forward problem
              2) the initial guess u if the forward problem is non-linear
          Note: p is not accessed
        - tol is the relative tolerance for the solution of the forward problem.
              [Default 1e-9].
        """
        self.problem.solveFwd(out, x, tol)

    
    def solveAdj(self, out, x, tol=1e-9):
        """
        Solve the linear adjoint problem.
        Parameters:
        - out: is the solution of the adjoint problem (i.e. the adjoint p) (Output parameter)
        - x = [u,a,p] provides
              1) the parameter variable a for assembling the adjoint operator
              2) the state variable u for assembling the adjoint right hand side
          Note: p is not accessed
        - tol is the relative tolerance for the solution of the adjoint problem.
              [Default 1e-9].
        """
        rhs = self.problem.generate_state()
        self.misfit.adj_rhs(x, rhs)
        self.problem.solveAdj(out, x, rhs, tol)
    
    def evalGradientParameter(self,x, mg):
        """
        Evaluate the gradient for the variational parameter equation at the point x=[u,a,p].
        Parameters:
        - x = [u,a,p] the point at which to evaluate the gradient.
        - mg the variational gradient (g, atest) being atest a test function in the parameter space
          (Output parameter)
        
        Returns the norm of the gradient in the correct inner product g_norm = sqrt(g,g)
        """ 
        self.prior.grad(x[PARAMETER], mg)
        tmp = self.generate_vector(PARAMETER)
        self.problem.eval_da(x, tmp)
        mg.axpy(1., tmp)
        self.prior.Msolver.solve(tmp, mg)
        #self.prior.Rsolver.solve(tmp, mg)
        return math.sqrt(mg.inner(tmp))
        
    
    def setPointForHessianEvaluations(self, x):
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        Parameters:
        - x = [u,a,p]: the point at which the Hessian or its Gauss-Newton approximation need to be evaluated.
        
        Note this routine should either:
        - simply store a copy of x and evaluate action of blocks of the Hessian on the fly
        - partially precompute the block of the hessian (if feasible)
        """
        self.problem.setLinearizationPoint(x)
        self.misfit.setLinearizationPoint(x)

        
    def solveFwdIncremental(self, sol, rhs, tol):
        """
        Solve the linearized (incremental) forward problem for a given rhs
        Parameters:
        - sol the solution of the linearized forward problem (Output)
        - rhs the right hand side of the linear system
        - tol the relative tolerance for the linear system
        """
        self.problem.solveIncremental(sol,rhs, False, tol)
        
    def solveAdjIncremental(self, sol, rhs, tol):
        """
        Solve the incremental adjoint problem for a given rhs
        Parameters:
        - sol the solution of the incremental adjoint problem (Output)
        - rhs the right hand side of the linear system
        - tol the relative tolerance for the linear system
        """
        self.problem.solveIncremental(sol,rhs, True, tol)
    
    def applyC(self, da, out):
        """
        Apply the C block of the Hessian to a (incremental) parameter variable.
        out = C da
        Parameters:
        - da the (incremental) parameter variable
        - out the action of the C block on da
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(ADJOINT,PARAMETER, da, out)
    
    def applyCt(self, dp, out):
        """
        Apply the transpose of the C block of the Hessian to a (incremental) adjoint variable.
        out = C^t dp
        Parameters:
        - dp the (incremental) adjoint variable
        - out the action of the C^T block on dp
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,ADJOINT, dp, out)

    
    def applyWuu(self, du, out, gn_approx=False):
        """
        Apply the Wuu block of the Hessian to a (incremental) state variable.
        out = Wuu du
        Parameters:
        - du the (incremental) state variable
        - out the action of the Wuu block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        self.misfit.apply_ij(STATE,STATE, du, out)
        if not gn_approx:
            tmp = self.generate_vector(STATE)
            self.problem.apply_ij(STATE,STATE, du, tmp)
            out.axpy(1., tmp)
    
    def applyWua(self, da, out):
        """
        Apply the Wua block of the Hessian to a (incremental) parameter variable.
        out = Wua da
        Parameters:
        - da the (incremental) parameter variable
        - out the action of the Wua block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(STATE,PARAMETER, da, out)
        tmp = self.generate_vector(STATE)
        self.misfit.apply_ij(STATE,PARAMETER, da, tmp)
        out.axpy(1., tmp)

    
    def applyWau(self, du, out):
        """
        Apply the Wau block of the Hessian to a (incremental) state variable.
        out = Wau du
        Parameters:
        - du the (incremental) state variable
        - out the action of the Wau block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(PARAMETER, STATE, du, out)
        tmp = self.generate_vector(PARAMETER)
        self.misfit.apply_ij(PARAMETER, STATE, du, tmp)
        out.axpy(1., tmp)
    
    def applyR(self, da, out):
        """
        Apply the regularization R to a (incremental) parameter variable.
        out = R da
        Parameters:
        - da the (incremental) parameter variable
        - out the action of R on da
        
        Note: this routine assumes that out has the correct shape.
        """
        self.prior.R.mult(da, out)
    
    def Rsolver(self):
        """
        Return an object Rsovler that is a suitable solver for the regularization
        operator R.
        
        The solver object should implement the method Rsolver.solve(z,r) such that
        R*z \\approx r.
        """
        return self.prior.Rsolver

    
    def applyRaa(self, da, out):
        """
        Apply the Raa block of the Hessian to a (incremental) parameter variable.
        out = Raa da
        Parameters:
        - da the (incremental) parameter variable
        - out the action of R on da
        
        Note: this routine assumes that out has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,PARAMETER, da, out)
        tmp = self.generate_vector(PARAMETER)
        self.misfit.apply_ij(PARAMETER,PARAMETER, da, tmp)
        out.axpy(1., tmp)
