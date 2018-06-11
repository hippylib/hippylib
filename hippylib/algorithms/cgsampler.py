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

from dolfin import Vector
from ..utils.random import parRandom
import math

class CGSampler:
    """ 
    This class implements the CG sampler algorithm to generate samples from :math:`\mathcal{N}(0, A^{-1})`.

    Reference:
        `Albert Parker and Colin Fox Sampling Gaussian Distributions in Krylov Spaces with Conjugate Gradient
        SIAM J SCI COMPUT, Vol 34, No. 3 pp. B312-B334`
    """

    def __init__(self):
        """
        Construct the solver with default parameters
        :code:`tolerance = 1e-4`
        
        :code:`print_level = 0`
        
        :code:`verbose = 0`
        """
        self.parameters = {}
        self.parameters["tolerance"] = 1e-4
        self.parameters["print_level"] = 0
        self.parameters["verbose"] = 0
        
        self.A = None
        self.converged = False
        self.iter = 0
        
        self.b = Vector()
        self.r = Vector()
        self.p = Vector()
        self.Ap = Vector()
                
    def set_operator(self, A):
        """
        Set the operator :code:`A`, such that :math:`x \sim \mathcal{N}(0, A^{-1})`.
        
        .. note:: :code:`A` is any object that provides the methods :code:`init_vector()` and :code:`mult()`
        
        """
        self.A = A
        self.A.init_vector(self.r,0)
        self.A.init_vector(self.p,0)
        self.A.init_vector(self.Ap,0)
        
        self.A.init_vector(self.b,0)
        parRandom.normal(1., self.b)
                        
    def sample(self, noise, s):
        """
        Generate a sample :math:`s ~ N(0, A^{-1})`.
        
        :code:`noise` is a :code:`numpy.array` of i.i.d. normal variables used as input.
        For a fixed realization of noise the algorithm is fully deterministic.
        The size of noise determine the maximum number of CG iterations.
        """
        s.zero()
        
        self.iter = 0
        self.converged = False
        
        # r0 = b
        self.r.zero()
        self.r.axpy(1., self.b)
        
        #p0 = r0
        self.p.zero()
        self.p.axpy(1., self.r)
        
        self.A.mult(self.p, self.Ap)
        
        d = self.p.inner(self.Ap)
        
        tol2 = self.parameters["tolerance"]*self.parameters["tolerance"]
        
        rnorm2_old = self.r.inner(self.r)
        
        if self.parameters["verbose"] > 0:
            print("initial residual = {0:g}".format( math.sqrt(rnorm2_old) ))
        
        while (not self.converged) and (self.iter < noise.shape[0]):
            gamma = rnorm2_old/d
            s.axpy(noise[self.iter]/math.sqrt(d), self.p)
            self.r.axpy(-gamma, self.Ap)
            rnorm2 = self.r.inner(self.r)
            beta = rnorm2/rnorm2_old
            # p_new = r + beta p
            self.p *= beta
            self.p.axpy(1., self.r)
            self.A.mult(self.p, self.Ap)
            d = self.p.inner(self.Ap)
            rnorm2_old = rnorm2
            
            if rnorm2 < tol2:
                self.converged = True
            else:
                rnorm2_old = rnorm2
                self.iter = self.iter+1
         
        if self.parameters["verbose"] > 0:       
            print("Final residual {0} after {1} iterations".format( math.sqrt(rnorm2_old), self.iter))
            
        
