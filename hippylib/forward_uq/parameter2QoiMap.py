# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

import numpy as np
import dolfin as dl
from ..modeling.variables import STATE, PARAMETER, ADJOINT
from ..utils.random import parRandom


class Parameter2QoiHessian:
    """
    This class implements matrix free application of the reduced hessian operator.
    
    The constructor takes the following parameters:
        - :code:`p2qoimap` - the object that describes the parameter-to-qoi map
    """
    def __init__(self, p2qoimap):
        """
        Construct the Hessian Operator of the parameter-to-qoi map
        """
        self.p2qoimap = p2qoimap
        self.ncalls = 0
        
        self.rhs_fwd = p2qoimap.generate_vector(STATE)
        self.rhs_adj = p2qoimap.generate_vector(ADJOINT)
        self.rhs_adj2 = p2qoimap.generate_vector(ADJOINT)
        self.uhat    = p2qoimap.generate_vector(STATE)
        self.phat    = p2qoimap.generate_vector(ADJOINT)
        self.yhelp = p2qoimap.generate_vector(PARAMETER)
    
    def init_vector(self, x, dim):
        """
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

            - :code:`x` - the vector to reshape
            - :code:`dim` - if 0 then :code:`x` will be reshaped to be compatible with the\
             range of the reduced Hessian, if 1 then :code:`x` will be reshaped to be \
             compatible with the domain of the reduced Hessian
                   
        .. note:: Since the reduced Hessian is a self adjoint operator, the range and \
        the domain is the same. Either way, we chose to add the parameter \
        :code:`dim` for consistency with the interface of :code:`dolfin.Matrix`.
        """
        self.p2qoimap.init_parameter(x)
        
    def mult(self,x,y):
        """
        Apply the Hessian of the parameter-to-qoi map to the vector :code:`x`
        Return the result in :code:`y`.
        """
        self.p2qoimap.applyC(x, self.rhs_fwd)
        self.p2qoimap.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.p2qoimap.applyWuu(self.uhat, self.rhs_adj)
        self.p2qoimap.applyWum(x, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)
        self.p2qoimap.solveAdjIncremental(self.phat, self.rhs_adj)
        self.p2qoimap.applyWmm(x, y)
        self.p2qoimap.applyCt(self.phat, self.yhelp)
        y.axpy(1., self.yhelp)
        self.p2qoimap.applyWmu(self.uhat, self.yhelp)
        y.axpy(-1., self.yhelp)

        
        self.ncalls += 1
    
    def inner(self,x,y):
        """
        Perform the inner product between :code:`x` and :code:`y` in the norm induced by the Hessian :math:`H`, i.e.
        :math:`(x, y)_H = x^T H y`
        """
        Ay = self.p2qoimap.generate_vector(PARAMETER)
        Ay.zero()
        self.mult(y,Ay)
        return x.inner(Ay)


class Parameter2QoiMap:
    def __init__(self, problem, qoi):
        """
        Create a parameter-to-qoi map given:

            - :code:`problem` - the description of the forward/adjoint problem and all the sensitivities
            - :code:`qoi` - the quantity of interest as a function of the state and parameter
        """
        self.problem = problem
        self.qoi = qoi
                
    def generate_vector(self, component = "ALL"):
        """
        By default, return the list :code:`[u, m, p]` where:

            - :code:`u` is any object that describes the state variable
            - :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
            (Need to support linear algebra operations)
            - :code:`p` is any object that describes the adjoint variable
        
        If :code:`component == STATE` return only :code:`u`

        If :code:`component == PARAMETER` return only :code:`m`
        
        If :code:`component == ADJOINT` return only :code:`p`
        """ 
        if component == "ALL":
            x = [self.problem.generate_state(), self.problem.generate_parameter(), self.problem.generate_state()]
        elif component == STATE:
            x = self.problem.generate_state()
        elif component == PARAMETER:
            x = self.problem.generate_parameter()
        elif component == ADJOINT:
            x = self.problem.generate_state()
            
        return x
    
    def init_parameter(self, m):
        """
        Reshape :code:`m` so that it is compatible with the parameter variable
        """
        self.problem.init_parameter(m)
            
    def eval(self, x):
        """
        Given the list :code:`x = [u, m, p]` which describes the state, parameter, and
        adjoint variable compute the QOI.
        
        .. note:: :code:`p` is not needed to compute the QOI
        """
        return self.qoi.eval(x)
        
    def solveFwd(self, out, x):
        """
        Solve the (possibly non-linear) forward problem.

        Parameters:

            - :code:`out` - is the solution of the forward problem (i.e. the state) (Output parameters)
            - :code:`x = [u, m, p]` provides

                1) the parameter variable :code:`m` for the solution of the forward problem
                2) the initial guess :code:`u` if the forward problem is non-linear
                
                .. note:: :code:`p` is not accessed
        """
        self.problem.solveFwd(out, x)

    
    def solveAdj(self, out, x):
        """
        Solve the linear adjoint problem.

        Parameters:

            - :code:`out` - is the solution of the adjoint problem (i.e. the adjoint p) (Output parameter)
            - :code:`x = [u, m, p]` provides

                1) the parameter variable :code:`m` for assembling the adjoint operator
                2) the state variable :code:`u` for assembling the adjoint right hand side

                .. note:: :code:`p` is not accessed
        """
        rhs = self.problem.generate_state()
        self.qoi.grad(STATE, x, rhs)
        rhs *= -1.
        self.problem.solveAdj(out, x, rhs)
    
    def evalGradientParameter(self,x, mg):
        """
        Evaluate the gradient for the variational parameter equation at the point :code:`x=[u, m, p]`.
        
        Parameters:

            - :code:`x = [u, m, p]` the point at which to evaluate the gradient.
            - :code:`mg` the variational gradient :math:`(g, mtest)` being mtest a test function in the parameter space \
            (Output parameter)
        """ 
        self.problem.evalGradientParameter(x, mg)
        tmp = self.problem.generate_parameter()
        self.qoi.grad(PARAMETER, x, tmp)
        mg.axpy(1., tmp)

        
    
    def setLinearizationPoint(self, x):
        """
        Specify the point :code:`x = [u, m, p]` at which the Hessian operator needs to be evaluated.
        
        Parameters:

            - :code:`x = [u, m, p]`: the point at which the Hessian needs to be evaluated.
        """
        self.problem.setLinearizationPoint(x, gauss_newton_approx=False)
        self.qoi.setLinearizationPoint(x)

        
    def solveFwdIncremental(self, sol, rhs):
        """
        Solve the linearized (incremental) forward problem for a given rhs
        
        Parameters:

            - :code:`sol` the solution of the linearized forward problem (Output)
            - :code:`rhs` the right hand side of the linear system
        """
        self.problem.solveIncremental(sol,rhs, False)
        
    def solveAdjIncremental(self, sol, rhs):
        """
        Solve the incremental adjoint problem for a given rhs
        
        Parameters:

            - :code:`sol` the solution of the incremental adjoint problem (Output)
            - :code:`rhs` the right hand side of the linear system
        """
        self.problem.solveIncremental(sol,rhs, True)
    
    def applyC(self, dm, out):
        """
        Apply the :math:`C` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`C dm`
        
        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`C` block on :code:dm
            
        .. note:: this routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(ADJOINT,PARAMETER, dm, out)
    
    def applyCt(self, dp, out):
        """
        Apply the transpose of the :math:`C` block of the Hessian to a (incremental) adjoint variable.
        :code:`out` = :math:`C^T dp`
        
        Parameters:

            - :code:`dp` the (incremental) adjoint variable
            - :code:`out` the action of the :math:`C^T` block on :code:`dp`
        
        .. note:: this routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,ADJOINT, dp, out)

    
    def applyWuu(self, du, out):
        """
        Apply the :math:`Wuu` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{uu} du`
        
        Parameters:

            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{uu}` block on :code:`du`
        
        .. note:: this routine assumes that :code:`out` has the correct shape.
        """
        self.qoi.apply_ij(STATE,STATE, du, out)
        tmp = self.generate_vector(STATE)
        self.problem.apply_ij(STATE,STATE, du, tmp)
        out.axpy(1., tmp)
    
    def applyWum(self, dm, out):
        """
        Apply the :math:`W_{um}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{um} dm`
        
        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{um}` block on :code:`du`
            
        .. note:: this routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(STATE,PARAMETER, dm, out)
        tmp = self.generate_vector(STATE)
        self.qoi.apply_ij(STATE,PARAMETER, dm, tmp)
        out.axpy(1., tmp)

    
    def applyWmu(self, du, out):
        """
        Apply the :math:`W_{mu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{mu} du`
        
        Parameters:

            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{mu}` block on :code:`du`
        
        .. note:: this routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(PARAMETER, STATE, du, out)
        tmp = self.generate_vector(PARAMETER)
        self.qoi.apply_ij(PARAMETER, STATE, du, tmp)
        out.axpy(1., tmp)
        
    def applyWmm(self, dm, out):
        """
        Apply the :math:`W_{mm}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{mm} dm`
        
        Parameters:
        
            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of :math:`W_{mm}` on :code:`dm`
        
        .. note:: this routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,PARAMETER, dm, out)
        tmp = self.generate_vector(PARAMETER)
        self.qoi.apply_ij(PARAMETER,PARAMETER, dm, tmp)
        out.axpy(1., tmp)
        
    def reduced_eval(self, m):
        """
        Evaluate the parameter-to-qoi map at a given realization :code:`m`
        
        .. note:: This evaluation requires the solution of a forward solve
        """
        u = self.problem.generate_state()
        if hasattr(self.problem, "initial_guess"):
            u.axpy(1., self.problem.initial_guess)
        self.problem.solveFwd(u, [u, m])
        return self.qoi.eval([u, m])
    
    def reduced_gradient(self, m, g):
        """
        Evaluate the gradient of parameter-to-qoi map at a given realization :code:`m`
        
        .. note:: This evaluation requires the solution of a forward and adjoint solve
        """
        u = self.problem.generate_state()
        p = self.problem.generate_state()
        self.solveFwd(u, [u, m, p])
        self.solveAdj(p, [u, m, p])
        self.evalGradientParameter(self, [u, m, p], g)
        return [u, m, p]
    
    def hessian(self, m=None, x=None):
        """
        Evaluate the Hessian of parameter-to-qoi map.
        
        If a relization of the parameter :code:`m` is given, this function will automatically
        compute the state :code:`u` and adjoint :code:`p`.
        
        As an alternative one can provide directly :code:`x = [u, m, p]`
        
        Returns an object of type :code:`ReducedHessianQOI` which provides the Hessian-apply functionality
        """
        if m is not None:
            assert x is None
            u = self.problem.generate_state()
            p = self.problem.generate_state()
            self.solveFwd(u, [u, m, p])
            self.solveAdj(p, [u, m, p])
            x = [u, m, p]
        else:
            assert x is not None
            
        self.setLinearizationPoint(x)
        return Parameter2QoiHessian(self)
        
def parameter2QoiMapVerify(rQOI, m0, h=None, eps=None, plotting = True,verbose = True):
    """
    Verify the gradient and the Hessian of a parameter-to-qoi map.
    It will produce two loglog plots of the finite difference checks
    for the gradient and for the Hessian.
    It will also check for symmetry of the Hessian.
    """
    rank = dl.MPI.rank(m0.mpi_comm())
    
    if h is None:
        h = rQOI.generate_vector(PARAMETER)
        parRandom.normal(1., h)

    
    x = rQOI.generate_vector()
    
    if hasattr(rQOI.problem, "initial_guess"):
        x[STATE].axpy(1., rQOI.problem.initial_guess)
    x[PARAMETER] = m0
    rQOI.solveFwd(x[STATE], x)
    rQOI.solveAdj(x[ADJOINT], x)
    qoi_x = rQOI.eval(x)
    
    grad_x = rQOI.generate_vector(PARAMETER)
    rQOI.evalGradientParameter(x, grad_x)
    grad_xh = grad_x.inner( h )
    
    H = rQOI.hessian(x=x)
    Hh = rQOI.generate_vector(PARAMETER)
    H.mult(h, Hh)
    
    if eps is None:
        n_eps = 32
        eps = np.power(.5, np.arange(n_eps-5,-5,-1))
    else:
        n_eps = eps.shape[0]
        
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)
    qois = np.zeros(n_eps)
    
    x_plus = rQOI.generate_vector()
    x_plus[STATE].axpy(1., x[STATE])
    
    for i in range(n_eps):
        my_eps = eps[i]
        
        x_plus[PARAMETER].zero()
        x_plus[PARAMETER].axpy(1., m0)
        x_plus[PARAMETER].axpy(my_eps, h)
        rQOI.solveFwd(x_plus[STATE],   x_plus)
        rQOI.solveAdj(x_plus[ADJOINT], x_plus)
        
        qoi_plus = rQOI.eval(x_plus)
        qois[i] = qoi_plus
        dQOI = qoi_plus - qoi_x
        err_grad[i] = abs(dQOI/my_eps - grad_xh)
        
        #Check the Hessian
        grad_xplus = rQOI.generate_vector(PARAMETER)
        rQOI.evalGradientParameter(x_plus, grad_xplus)
        
        err  = grad_xplus - grad_x
        err *= 1./my_eps
        err -= Hh
        
        err_H[i] = err.norm('linf')

        if rank == 0 and verbose:
            print( "{0:1.7e} {1:1.7e} {2:1.7e} {3:1.7e}".format(eps[i], qois[i], err_grad[i], err_H[i]))
    
    if plotting and (rank == 0):
        parameter2QoiMapVerifyPlotErrors(eps, err_grad, err_H) 

    fd_check = np.zeros((eps.shape[0], 4))
    fd_check[:,0] = eps
    fd_check[:,1] = qois
    fd_check[:,2] = err_grad
    fd_check[:,3] = err_H
    
    if rank == 0:
        np.savetxt('fd_check.txt', fd_check)

    out = {}
    out['err_grad'] = err_grad
    out['err_H'] = err_H

    # Compute symmetry error  
    xx = rQOI.generate_vector(PARAMETER)
    parRandom.normal(1., xx)
    yy = rQOI.generate_vector(PARAMETER)
    parRandom.normal(1., yy)
    
    ytHx = H.inner(yy,xx)
    xtHy = H.inner(xx,yy)
    rel_sym_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    out['rel_sym_error'] = rel_sym_error
    if rank ==  0 and verbose:
        print( "(yy, H xx) - (xx, H yy) = ", rel_sym_error)
        if(rel_sym_error > 1e-10):
            print( "HESSIAN IS NOT SYMMETRIC!!")
        
    return out

def parameter2QoiMapVerifyPlotErrors(eps, err_grad, err_H):
    try:
        import matplotlib.pyplot as plt
    except:
        print( "Matplotlib is not installed.")
        return
    
    plt.figure()
    plt.subplot(121)
    plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
    plt.title("FD Gradient Check")
    plt.subplot(122)
    plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
    plt.title("FD Hessian Check")
