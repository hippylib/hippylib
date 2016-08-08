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

import numpy as np
import matplotlib.pyplot as plt

from variables import STATE, PARAMETER, ADJOINT
from reducedHessian import ReducedHessian

class ModelTemplate:
    """
    This class is a template for all the methods that a model object should
    provide.
    In the following we will denote with
    - u the state variable
    - a the parameter variable
    - p the adjoint variable
    
    For a concrete example see application/poisson/model.py.
    
    """
    
    def __init__(self, mesh, Vh, prior, args):
        """
        Construct a model by proving a mesh, the finite element spaces
        for the STATE/ADJOINT variable and the PARAMETER variable, and a
        model for the prior information/regularization.
        Pass any other parameter as needed.
        """
                
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
        return [None, None, None] #[u,a,p]
    
    def init_parameter(self, a):
        """
        Reshape a so that it is compatible with the parameter variable
        """
        return
            
    def cost(self, x):
        """
        Given the list x = [u,a,p] which describes the state, parameter, and
        adjoint variable compute the cost functional as the sum of 
        the misfit functional and the regularization functional.
        
        Return the list [cost functional, regularization functional, misfit functional]
        
        Note: p is not needed to compute the cost functional
        """
        return [None, None, None] #[cost, reg, misfit]
    
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
        return

    
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
        return
    
    def evalGradientParameter(self,x, mg):
        """
        Evaluate the gradient for the variational parameter equation at the point x=[u,a,p].
        Parameters:
        - x = [u,a,p] the point at which to evaluate the gradient.
        - mg the variational gradient (g, atest) being atest a test function in the parameter space
          (Output parameter)
        
        Returns the norm of the gradient in the correct inner product g_norm = sqrt(g,g)
        """ 
        return None #gradient norm
        
    
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
        return

        
    def solveFwdIncremental(self, sol, rhs, tol):
        """
        Solve the linearized (incremental) forward problem for a given rhs
        Parameters:
        - sol the solution of the linearized forward problem (Output)
        - rhs the right hand side of the linear system
        - tol the relative tolerance for the linear system
        """
        return
        
    def solveAdjIncremental(self, sol, rhs, tol):
        """
        Solve the incremental adjoint problem for a given rhs
        Parameters:
        - sol the solution of the incremental adjoint problem (Output)
        - rhs the right hand side of the linear system
        - tol the relative tolerance for the linear system
        """
        return
    
    def applyC(self, da, out):
        """
        Apply the C block of the Hessian to a (incremental) parameter variable.
        out = C da
        Parameters:
        - da the (incremental) parameter variable
        - out the action of the C block on da
        
        Note: this routine assumes that out has the correct shape.
        """
        return
    
    def applyCt(self, dp, out):
        """
        Apply the transpose of the C block of the Hessian to a (incremental) adjoint variable.
        out = C^t dp
        Parameters:
        - dp the (incremental) adjoint variable
        - out the action of the C^T block on dp
        
        Note: this routine assumes that out has the correct shape.
        """
        return

    
    def applyWuu(self, du, out):
        """
        Apply the Wuu block of the Hessian to a (incremental) state variable.
        out = Wuu du
        Parameters:
        - du the (incremental) state variable
        - out the action of the Wuu block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        return
    
    def applyWua(self, da, out):
        """
        Apply the Wua block of the Hessian to a (incremental) parameter variable.
        out = Wua da
        Parameters:
        - da the (incremental) parameter variable
        - out the action of the Wua block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        return

    
    def applyWau(self, du, out):
        """
        Apply the Wau block of the Hessian to a (incremental) state variable.
        out = Wau du
        Parameters:
        - du the (incremental) state variable
        - out the action of the Wau block on du
        
        Note: this routine assumes that out has the correct shape.
        """
        return
    
    def applyR(self, da, out):
        """
        Apply the regularization R to a (incremental) parameter variable.
        out = R da
        Parameters:
        - da the (incremental) parameter variable
        - out the action of R on da
        
        Note: this routine assumes that out has the correct shape.
        """
        return
    
    def Rsolver(self):
        """
        Return an object Rsovler that is a suitable solver for the regularization
        operator R.
        
        The solver object should implement the method Rsolver.solve(z,r) such that
        R*z \\approx r.
        """

    
    def applyRaa(self, da, out):
        """
        Apply the Raa block of the Hessian to a (incremental) parameter variable.
        out = Raa da
        Parameters:
        - da the (incremental) parameter variable
        - out the action of R on da
        
        Note: this routine assumes that out has the correct shape.
        """
        return
    
def modelVerify(model,a0, innerTol, is_quadratic = False):
    """
    Verify the reduced Gradient and the Hessian of a model.
    It will produce two loglog plots of the finite difference checks
    for the gradient and for the Hessian.
    It will also check for symmetry of the Hessian.
    """
    
    h = model.generate_vector(PARAMETER)
    h.set_local(np.random.normal(0, 1, len( h.array() )) )
    
    x = model.generate_vector()
    x[PARAMETER] = a0
    model.solveFwd(x[STATE], x, innerTol)
    model.solveAdj(x[ADJOINT], x, innerTol)
    cx = model.cost(x)
    
    grad_x = model.generate_vector(PARAMETER)
    model.evalGradientParameter(x, grad_x)
    grad_xh = grad_x.inner( h )
    
    model.setPointForHessianEvaluations(x)
    H = ReducedHessian(model,innerTol)
    Hh = model.generate_vector(PARAMETER)
    H.mult(h, Hh)
    
    n_eps = 32
    eps = np.power(.5, np.arange(n_eps))
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)
    
    for i in range(n_eps):
        my_eps = eps[i]
        
        x_plus = model.generate_vector()
        x_plus[PARAMETER].set_local( a0.array() )
        x_plus[PARAMETER].axpy(my_eps, h)
        model.solveFwd(x_plus[STATE],   x_plus, innerTol)
        model.solveAdj(x_plus[ADJOINT], x_plus,innerTol)
        
        dc = model.cost(x_plus)[0] - cx[0]
        err_grad[i] = abs(dc/my_eps - grad_xh)
        
        #Check the Hessian
        grad_xplus = model.generate_vector(PARAMETER)
        model.evalGradientParameter(x_plus, grad_xplus)
        
        err  = grad_xplus - grad_x
        err *= 1./my_eps
        err -= Hh
        
        err_H[i] = err.norm('linf')
    
    if is_quadratic:
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps[0], err_H[0], "-ob", [10*eps[0], eps[0], 0.1*eps[0]], [err_H[0],err_H[0],err_H[0]], "-.k")
        plt.title("FD Hessian Check")
    else:  
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
        plt.title("FD Hessian Check")
    
        
    xx = model.generate_vector(PARAMETER)
    xx.set_local( np.random.normal(0, 1, len( xx.array() )) )
    yy = model.generate_vector(PARAMETER)
    yy.set_local( np.random.normal(0, 1, len( yy.array() )) )
    
    ytHx = H.inner(yy,xx)
    xtHy = H.inner(xx,yy)
    rel_symm_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    print "(yy, H xx) - (xx, H yy) = ", rel_symm_error
    if(rel_symm_error > 1e-10):
        print "HESSIAN IS NOT SYMMETRIC!!"
