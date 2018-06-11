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

from .variables import STATE, PARAMETER, ADJOINT
from .reducedHessian import ReducedHessian
from ..utils.random import parRandom
    
def modelVerify(model,m0, innerTol, is_quadratic = False, misfit_only=False, verbose = True, eps = None):
    """
    Verify the reduced Gradient and the Hessian of a model.
    It will produce two loglog plots of the finite difference checks for the gradient and for the Hessian.
    It will also check for symmetry of the Hessian.
    """
    if misfit_only:
        index = 2
    else:
        index = 0
    
    h = model.generate_vector(PARAMETER)
    parRandom.normal(1., h)

    
    x = model.generate_vector()
    x[PARAMETER] = m0
    model.solveFwd(x[STATE], x, innerTol)
    model.solveAdj(x[ADJOINT], x, innerTol)
    cx = model.cost(x)
    
    grad_x = model.generate_vector(PARAMETER)
    model.evalGradientParameter(x, grad_x,misfit_only=misfit_only)
    grad_xh = grad_x.inner( h )
    
    model.setPointForHessianEvaluations(x)
    H = ReducedHessian(model,innerTol, misfit_only=misfit_only)
    Hh = model.generate_vector(PARAMETER)
    H.mult(h, Hh)
    
    if eps is None:
        n_eps = 32
        eps = np.power(.5, np.arange(n_eps))
        eps = eps[::-1]
    else:
        n_eps = eps.shape[0]
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)
    
    for i in range(n_eps):
        my_eps = eps[i]
        
        x_plus = model.generate_vector()
        x_plus[PARAMETER].axpy(1., m0 )
        x_plus[PARAMETER].axpy(my_eps, h)
        model.solveFwd(x_plus[STATE],   x_plus, innerTol)
        model.solveAdj(x_plus[ADJOINT], x_plus,innerTol)
        
        dc = model.cost(x_plus)[index] - cx[index]
        err_grad[i] = abs(dc/my_eps - grad_xh)
        
        #Check the Hessian
        grad_xplus = model.generate_vector(PARAMETER)
        model.evalGradientParameter(x_plus, grad_xplus,misfit_only=misfit_only)
        
        err  = grad_xplus - grad_x
        err *= 1./my_eps
        err -= Hh
        
        err_H[i] = err.norm('linf')
    
    if verbose:
        modelVerifyPlotErrors(is_quadratic, eps, err_grad, err_H)

    xx = model.generate_vector(PARAMETER)
    parRandom.normal(1., xx)
    yy = model.generate_vector(PARAMETER)
    parRandom.normal(1., yy)
    
    ytHx = H.inner(yy,xx)
    xtHy = H.inner(xx,yy)
    if np.abs(ytHx + xtHy) > 0.: 
        rel_symm_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    else:
        rel_symm_error = abs(ytHx - xtHy)
    if verbose:
        print( "(yy, H xx) - (xx, H yy) = ", rel_symm_error)
        if rel_symm_error > 1e-10:
            print( "HESSIAN IS NOT SYMMETRIC!!")
            
    return eps, err_grad, err_H

def modelVerifyPlotErrors(is_quadratic, eps, err_grad, err_H):
    try:
        import matplotlib.pyplot as plt
    except:
        print( "Matplotlib is not installed.")
        return
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