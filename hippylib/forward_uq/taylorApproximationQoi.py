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
from ..modeling.variables import STATE, PARAMETER, ADJOINT
from ..algorithms.randomizedEigensolver import doublePassG

class TaylorApproximationQoi:
    """
    This class computes the first and second order Taylor approximation of the parameter-to-qoi map.
    It provides methods to evaluate the Taylor approximation for a specific realization of the parameter
    and to analytically compute Expectation and Variance of the Taylor approximation
    with respect to a Gaussian propability distribution.
    """
    def __init__(self, p2qoimap, distribution):
        """
        Constructor:

            - :code:`p2qoimap` - an object of type :code:`ReducedQOI` that describes the parameter-to-qoi map
            - :code:`distribution` - an object of type :code:`hIPPYlib._Prior` that describes the prior Gaussian distribution.
        """
        self.p2qoimap = p2qoimap
        self.distribution = distribution
        
        self.x_bar = self.p2qoimap.generate_vector()
        self.x_bar[PARAMETER].axpy(1., distribution.mean)
        
        if hasattr(self.p2qoimap.problem, "initial_guess"):
            self.x_bar[STATE].axpy(1., self.p2qoimap.problem.initial_guess)   
        
        self.q_bar = 0.
        self.g_bar =  self.p2qoimap.generate_vector(PARAMETER)
        
        self.d = None
        self.U = None
        self.H = None
        
    def computeLowRankFactorization(self, Omega):
        """
        Compute the LowRank Factorization of the prior-preconditioned Hessian of the
        parameter-to-qoi map.
        
        The trace of the the prior-preconditioned Hessian is also computed as a post-process of the
        LowRank factorization.
        
        This method needs to be called before trying to compute the moments of the the quadratic Taylor
        approx of the parameter-to-qoi map.
        
        Inputs:

            - :code:`Omega` - Gaussian random matrix for randomized method
            - :code:`k` - the number of eigenpairs to be retained in the LowRank factorization of the prior-preconditioned Hessian.
        """
        
        self.p2qoimap.solveFwd(self.x_bar[STATE], self.x_bar)
        self.p2qoimap.solveAdj(self.x_bar[ADJOINT], self.x_bar)
        
        self.q_bar = self.p2qoimap.eval(self.x_bar)
        self.p2qoimap.evalGradientParameter(self.x_bar, self.g_bar)
        
        self.H = self.p2qoimap.hessian(x=self.x_bar)
        if hasattr(self.distribution, "R"):
            self.d, self.U = doublePassG(self.H, self.distribution.R, self.distribution.Rsolver, Omega, Omega.nvec())
        else:
            self.d, self.U = doublePassG(self.H, self.distribution.Hlr, self.distribution.Hlr, Omega, Omega.nvec())
            
        
    def expectedValue(self, order=2):
        """
        Returns the expected value (computed analytically) of the qoi with respect to a Gaussian distribution
        for the parameter.
        
        Input:

            - :code:`order` - is the order of the Taylor approximation, currently 1 (linear) or 2 (quadratic)
        """
        if order == 1:
            correction = 0.
        elif order == 2:
            correction = .5*np.sum(self.d)
        else:
            raise
            
        return self.q_bar + correction
    
    def variance(self, order=2):
        """
        Returns the variance (computed analytically) of the qoi with respect to a Gaussian distribution
        for the parameter.
        
        Input:

            - :code:`order` - is the order of the Taylor approximation, currently 1 (linear) or 2 (quadratic)
        """
        Rinv_g = self.p2qoimap.generate_vector(PARAMETER)
        
        if hasattr(self.distribution, "R"):
            self.distribution.Rsolver.solve(Rinv_g, self.g_bar)
        else:
            self.distribution.Hlr.solve(Rinv_g, self.g_bar)
            
        lin_var = Rinv_g.inner(self.g_bar)
        if order == 1:
            correction = 0.
        elif order == 2:
            correction =.5*np.sum(np.power(self.d,2))
        else:
            raise
        
        return lin_var + correction
    
    def eval(self, m, order=2):
        """
        Evaluates the Taylor approx of the qoi for a given realization of the parameter.
        
        Input:
        
            - :code:`m` - a specific realization of the uncertain parameter 
            - :code:`order` - is the order of the Taylor approximation, currently 1 (linear) or 2 (quadratic)
        """
        dm = m - self.x_bar[PARAMETER]
        if order == 1:
            correction = 0.
        elif order == 2:
            correction = .5*self.H.inner(dm,dm)
        else:
            raise
        return self.q_bar + self.g_bar.inner(dm) + correction
                
def plotEigenvalues(d):
    """
    Plots the eigenvalues d in a semilogy scale.
    Positive eigenvalues are marked in blue, negative eigenvalues are marked in red.
    """
    try:
        import matplotlib.pyplot as plt
    except:
        print( "Matplotlib is not installed.")
        return
        
    d_abs = np.abs(d)
    index = np.argsort(d_abs)[::-1]
    inv_index = np.zeros(index.shape, dtype=index.dtype)
    inv_index[index] = np.arange(index.shape[0])
    plt.figure()
    plt.semilogy(inv_index[d>0], d_abs[d>0], '*b', label="positive" )
    plt.semilogy(inv_index[d<0], d_abs[d<0], '*r', label="negative" )
    plt.legend()
    plt.xlabel("eigenvalues index")
    plt.ylabel("abs(eigenvalues)")

        
        

        
