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

from dolfin import Vector, mpi_comm_world
from ..utils.random import parRandom
import math
from .linalg import Solver2Operator

def rademacher_engine(v):
    """
    Generate a vector of :math:`n` i.i.d. Rademacher variables.
    """
    parRandom.rademacher(v)
    
def gaussian_engine(v):
    """
    Generate a vector of :math:`n` i.i.d. standard normal variables.
    """
    parRandom.normal(1., v)

class TraceEstimator:
    """
    An unbiased stochastic estimator for the trace of :math:`A,\\, d = \\sum_{j=1}^k (v_j, A v_j)`, where

    - :math:`v_j` are i.i.d. Rademacher or Gaussian random vectors.
    - :math:`(\\cdot,\\cdot)` represents the inner product.
    
    The number of samples :math:`k` is estimated at run time based on the variance of the estimator.

    Reference: Haim Avron and Sivan Toledo, Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix,
    Journal of the ACM (JACM), 58 (2011), p. 17.
    """
    def __init__(self, A, solve_mode=False, accurancy = 1e-1, init_vector=None, random_engine=rademacher_engine, mpi_comm=mpi_comm_world()):
        """
        Constructor:

        - :code:`A`:             an operator
        - :code:`solve_mode`:    if :code:`True` we estimate the trace of :code:`A`:math:`^{-1}`, otherwise of :code:`A`.
        - code:`accurancy`:     we stop when the standard deviation of the estimator is less then
                         :code:`accurancy`*tr(:code:`A`).
        - :code:`init_vector`:   use a custom function to initialize a vector compatible with the
                         range/domain of :code:`A`.
        - :code:`random_engine`: which type of i.i.d. random variables to use (Rademacher or Gaussian). 
        """
        if solve_mode:
            self.A = Solver2Operator(A)
        else:
            self.A = A
        self.accurancy = accurancy
        self.random_engine = random_engine
        self.iter = 0
        
        self.z = Vector(mpi_comm)
        self.Az = Vector(mpi_comm)
        
        if init_vector is None:
            A.init_vector(self.z, 0)
            A.init_vector(self.Az, 0)
        else:
            init_vector(self.z, 0)
            init_vector(self.Az, 0)
            
    def __call__(self, min_iter=5, max_iter=100):
        """
        Estimate the trace of :code:`A` (or :code:`A`:math:`^-1`) using at least
        :code:`min_iter` and at most :code:`max_iter` samples.
        """
        sum_tr = 0
        sum_tr2 = 0
        self.iter = 0
        
        while self.iter < min_iter:
            self.iter += 1
            self.random_engine(self.z)
            self.A.mult(self.z, self.Az)
            tr = self.z.inner(self.Az)
            sum_tr += tr
            sum_tr2 += tr*tr
            
        exp_tr = sum_tr / float(self.iter)
        exp_tr2 = sum_tr2 / float(self.iter)
        var_tr = exp_tr2 - exp_tr*exp_tr
        
#        print( exp_tr, math.sqrt( var_tr ), self.accurancy*exp_tr)
        
        self.converged = True
        while (math.sqrt( var_tr ) > self.accurancy*exp_tr):
            self.iter += 1
            self.random_engine(self.z)
            self.A.mult(self.z, self.Az)
            tr = self.z.inner(self.Az)
            sum_tr += tr
            sum_tr2 += tr*tr
            exp_tr = sum_tr / float(self.iter)
            exp_tr2 = sum_tr2 / float(self.iter)
            var_tr = exp_tr2 - exp_tr*exp_tr
#            print( exp_tr, math.sqrt( var_tr ), self.accurancy*exp_tr)
            if (self.iter > max_iter):
                self.converged = False
                break
            
        return exp_tr, var_tr
    
        