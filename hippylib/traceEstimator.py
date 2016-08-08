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

from dolfin import Vector
import numpy as np
import math

def rademacher_engine(n):
    """
    Generate a vector of n i.i.d. Rademacher variables.
    """
    omega = np.random.rand(n)
    omega[omega < .5 ] = -1.
    omega[omega >= .5 ] = 1.
    return omega
    
def gaussian_engine(n):
    """
    Generate a vector of n i.i.d. standard normal variables.
    """
    return np.random.randn(n)

class TraceEstimator:
    """
    An unbiased stochastic estimator for the trace of A.
    d = \sum_{j=1}^k (vj, A vj)
    where
    - vj are i.i.d. Rademacher or Gaussian random vectors
    - (.,.) represents the inner product
    
    The number of samples k is estimated at run time based on
    the variance of the estimator.

    REFERENCE:
    
    Haim Avron and Sivan Toledo,
    Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix,
    Journal of the ACM (JACM), 58 (2011), p. 17.
    """
    def __init__(self, A, solve_mode=False, accurancy = 1e-1, init_vector=None, random_engine=rademacher_engine):
        """
        Constructor:
        - A: an operator
        - solve_mode:    if True we estimate the trace of A^{-1}, otherwise of A.
        - accurancy:     we stop when the standard deviation of the estimator is less then
                         accurancy*tr(A).
        - init_vector:   use a custom function to initialize a vector compatible with the
                         range/domain of A
        - random_engine: which type of i.i.d. random variables to use (Rademacher or Gaussian)  
        """
        self.A = A
        self.accurancy = accurancy
        self.random_engine = random_engine
        self.iter = 0
        
        self.z = Vector()
        self.Az = Vector()
        
        if solve_mode:
            self._apply = self._apply_solve
        else:
            self._apply = self._apply_mult
        
        if init_vector is None:
            A.init_vector(self.z, 0)
            A.init_vector(self.Az, 0)
        else:
            init_vector(self.z, 0)
            init_vector(self.Az, 0)
            
    def _apply_mult(self, z, Az):
        self.A.mult(z, Az)
        
    def _apply_solve(self, z, Az):
        self.A.solve(Az, z)
        
    def __call__(self, min_iter=5, max_iter=100):
        """
        Estimate the trace of A (or A^-1) using at least
        min_iter and at most max_iter samples.
        """
        sum_tr = 0
        sum_tr2 = 0
        self.iter = 0
        size = len(self.z.array())
        
        while self.iter < min_iter:
            self.iter += 1
            self.z.set_local(self.random_engine(size))
            self._apply(self.z, self.Az)
            tr = self.z.inner(self.Az)
            sum_tr += tr
            sum_tr2 += tr*tr
            
        exp_tr = sum_tr / float(self.iter)
        exp_tr2 = sum_tr2 / float(self.iter)
        var_tr = exp_tr2 - exp_tr*exp_tr
        
#        print exp_tr, math.sqrt( var_tr ), self.accurancy*exp_tr
        
        self.converged = True
        while (math.sqrt( var_tr ) > self.accurancy*exp_tr):
            self.iter += 1
            self.z.set_local(self.random_engine(size))
            self._apply(self.z, self.Az)
            tr = self.z.inner(self.Az)
            sum_tr += tr
            sum_tr2 += tr*tr
            exp_tr = sum_tr / float(self.iter)
            exp_tr2 = sum_tr2 / float(self.iter)
            var_tr = exp_tr2 - exp_tr*exp_tr
#            print exp_tr, math.sqrt( var_tr ), self.accurancy*exp_tr
            if (self.iter > max_iter):
                self.converged = False
                break
            
        return exp_tr, var_tr
    
        