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

import dolfin as dl
import numpy as np
from ..modeling.variables import STATE, PARAMETER, ADJOINT
from ..utils.random import parRandom


class Qoi(object):
    """
    Abstract class to model the Quantity of Interest.
    In the following :code:`x` will denote the variable :code:`[u, m, p]`, denoting respectively
    the state :code:`u`, the parameter :code:`m`, and the adjoint variable :code:`p`.
    
    The methods in the class QOI will usually access the state :code:`u` and possibly the
    parameter :code:`m`. 
    """        
    def eval(self, x):
        """
        Given :code:`x` evaluate the cost functional.
        Only the state :code:`u` and (possibly) the parameter :code:`m` are accessed.
        """
        raise NotImplementedError("Child class should implement method eval")
        return 0

    def grad(self,i, x,g):
        """
        Evaluate the gradient with respect to the state.
        Only the state :code:`u` and (possibly) the parameter :code:`m` are accessed. 
        """
        raise NotImplementedError("Child class should implement method grad")

    def setLinearizationPoint(self, x):
        raise NotImplementedError("Child class should implement method setLinearizationPoint")

    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """
        raise NotImplementedError("Child class should implement method apply_ij")
        
        
#FD check (STATE)
def qoiVerify(qoi, x, generate_state, h=None, plotting = True):
    
    rank = dl.MPI.rank(x[STATE].mpi_comm())
    
    if h is None:
        h = generate_state()
        parRandom.normal(1., h)

    qoi_x = qoi.eval(x)
    grad_x = generate_state()
    qoi.setLinearizationPoint(x)
    qoi.grad(STATE, x,grad_x)
    grad_xh = grad_x.inner(h)
    Hh = generate_state()
    qoi.apply_ij(STATE,STATE, h, Hh)
    
    n_eps = 32
    eps = np.power(.5, np.arange(n_eps, 0, -1))
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)
    
    for i in range(n_eps):
        my_eps = eps[i]
        
        state_plus = generate_state()
        state_plus.axpy(1., x[STATE])
        state_plus.axpy(my_eps, h)
        
        dq = qoi.eval([state_plus, x[PARAMETER], x[ADJOINT]] ) - qoi_x
        err_grad[i] = abs(dq/my_eps - grad_xh)
        
        grad_xplus = generate_state()
        qoi.grad(STATE, [state_plus, x[PARAMETER], x[ADJOINT]], grad_xplus)
        
        err  = grad_xplus - grad_x
        err *= 1./my_eps
        err -= Hh
        
        err_H[i] = err.norm('linf')
    
    if plotting and (rank==0):
        qoiVerifyPlotErrors(eps, err_grad, err_H)
        
    return eps, err_grad, err_H
        

def qoiVerifyPlotErrors(eps, err_grad, err_H):
    try:
        import matplotlib.pyplot as plt
    except:
        print( "Matplotlib is not installed.")
        return
    plt.figure()
    plt.subplot(121)
    try:
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
    except:
        plt.semilogx(eps, err_grad, "-ob")
    plt.title("FD Gradient Check")
    plt.subplot(122)
    try:
        plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
    except:
        pass
    plt.title("FD Hessian Check")
