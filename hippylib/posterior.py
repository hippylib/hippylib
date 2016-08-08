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

from dolfin import Vector, Function, File
from lowRankOperator import LowRankOperator
import numpy as np

class LowRankHessian:
    """
    Operator that represents the action of the low rank approx
    of the Hessian and of its inverse.
    """
    def __init__(self, prior, d, U):
        self.prior = prior
        self.LowRankH = LowRankOperator(d, U)
        dsolve = d / (np.ones(d.shape, dtype=d.dtype) + d)
        self.LowRankHinv = LowRankOperator(dsolve, U)
        self.help = Vector()
        self.init_vector(self.help, 0)
        self.help1 = Vector()
        self.init_vector(self.help1, 0)
        
    def init_vector(self,x, dim):
        self.prior.init_vector(x,dim)
    
    def inner(self,x,y):
        Hx = Vector()
        self.init_vector(Hx, 0)
        self.mult(x, Hx)
        return Hx.inner(y)
        
    def mult(self, x, y):
        self.prior.R.mult(x,y)
        self.LowRankH.mult(y, self.help)
        self.prior.R.mult(self.help,self.help1)
        y.axpy(1, self.help1)
        
        
    def solve(self, sol, rhs):
        self.prior.Rsolver.solve(sol, rhs)
        self.LowRankHinv.mult(rhs, self.help)
        sol.axpy(-1, self.help)
        
class LowRankPosteriorSampler:
    """
    Object to sample from the low-rank approximation
    of the posterior.
    y = ( I - U S U^TR) x,
    where
    S = I - (I + D)^{-1/2},
    x ~ N(0, R^{-1})
    """
    def __init__(self, prior, d, U):
        self.prior = prior        
        ones = np.ones( d.shape, dtype=d.dtype )        
        self.d = ones - np.power(ones + d, -.5)
        self.lrsqrt = LowRankOperator(self.d, U)
        self.help = Vector()
        self.init_vector(self.help, 0)
        
    def init_vector(self,x, dim):
        self.prior.init_vector(x,dim)
        
    def sample(self, noise, s):
        self.prior.R.mult(noise, self.help)
        self.lrsqrt.mult(self.help, s)
        s.axpy(-1, noise)

class GaussianLRPosterior:
    """
    Class for the low rank Gaussian Approximation of the Posterior.
    This class provides functionality for approximate Hessian
    apply, solve, and Gaussian sampling based on the low rank
    factorization of the Hessian.
    
    In particular if d and U are the dominant eigenpairs of
    H_misfit U[:,i] = d[i] R U[:,i]
    then we have:
    
    - low-rank Hessian apply:
      y = ( R + RU D U^TR) x
      
    - low-rank Hessian solve:
      y = (R^-1 - U (I + D^{-1})^{-1} U^T) x
      
    - low-rank Hessian Gaussian sampling:
      y = ( I - U S U^TR) x, where S = I - (I + D)^{-1/2} and x ~ N(0, R^{-1})
    """
    def __init__(self, prior, d, U, mean=None):
        """
        Construct the Gaussian approximation of the posterior.
        Input:
        - prior: the prior mode
        - d:     the dominant generalized eigenvalues of the Hessian misfit
        - U:     the dominant generalized eigenvector of the Hessian misfit U^T R U = I.
        - mean:  the MAP point
        """
        self.prior = prior
        self.d = d
        self.U = U
        self.Hlr = LowRankHessian(prior, d, U)
        self.sampler = LowRankPosteriorSampler(self.prior, self.d, self.U)
        self.mean=None
        
    def init_vector(self,x, dim):
        """
        Inizialize a vector x to be compatible with the range/domain of H.
        If dim == "noise" inizialize x to be compatible with the size of
        white noise used for sampling.
        """
        self.prior.init_vector(x,dim)
        
    def sample(self, *args, **kwargs):
        """
        possible calls:
        
        1) sample(s_prior, s_post, add_mean=True)
           Given a prior sample  s_prior compute a sample s_post from the posterior
           - s_prior is a sample from the prior centered at 0 (input)
           - s_post is a sample from the posterior (output)
           - if add_mean=True (default) than the samples will be centered at the map
             point
             
        2) sample(noise, s_prior, s_post, add_mean=True)
           Given a noise ~ N(0, I) compute a sample s_prior from the prior and s_post from the posterior
           - noise is a realization of white noise (input)
           - s_prior is a sample from the prior (output)
           - s_post  is a sample from the posterior
           - if add_mean=True (default) than the prior and posterior samples will be
                centered at the respective means.
        """
        add_mean = True
        for name, value in kwargs.items():
            if name == "add_mean":
                add_mean = value
            else:
                raise NameError(name)
        
        if len(args) == 2:
            self._sample_given_prior(args[0], args[1])
            if add_mean:
                args[1].axpy(1., self.mean)
        elif len(args) == 3:
            self._sample_given_white_noise(args[0], args[1], args[2])
            if add_mean:
                    args[1].axpy(1., self.prior.mean) 
                    args[2].axpy(1., self.mean)
        else:
            raise NameError('Invalid number of parameters in Posterior::sample')
        
    def _sample_given_white_noise(self, noise, s_prior, s_post):
        self.prior.sample(noise, s_prior, add_mean=False)
        self.sampler.sample(s_prior, s_post)
        
    def _sample_given_prior(self,s_prior, s_post):
        self.sampler.sample(s_prior, s_post)
    
    def exportU(self, Vh, fname, varname = "evect", normalize=1):
        """
        Export in paraview the generalized eigenvectors U.
        Inputs:
        - Vh:        the parameter finite element space
        - fname:     the name of the paraview output file
        - varname:   the name of the paraview variable
        - normalize: if True the eigenvector are rescaled such that || u ||_inf = 1 
        """
        evect = Function(Vh, name=varname)
        fid = File(fname)
        
        for i in range(0,self.U.shape[1]):
            Ui = self.U[:,i]
            if normalize:
                s = 1/np.linalg.norm(Ui, np.inf)
                evect.vector().set_local(s*Ui)
            else:
                evect.vector().set_local(Ui)
            fid << evect
            
    def trace(self, method="Exact", tol=1e-1, min_iter=20, max_iter=100):
        """
        Compute/Estimate the trace of the posterior, prior distribution
        and the trace of the data informed correction.
        
        See _Prior.trace for more details.
        """
        pr_trace = self.prior.trace(method, tol, min_iter, max_iter)
        corr_trace = self.Hlr.LowRankHinv.trace(self.prior.M)
        post_trace = pr_trace - corr_trace
        return post_trace, pr_trace, corr_trace
    
    def pointwise_variance(self, method="Exact", path_len = 8):
        """
        Compute/Estimate the pointwise variance of the posterior, prior distribution
        and the pointwise variance reduction informed by the data.
        
        See _Prior.pointwise_variance for more details. 
        """
        pr_pointwise_variance = self.prior.pointwise_variance(method, path_len)
        correction_pointwise_variance = Vector()
        self.init_vector(correction_pointwise_variance, 0)
        self.Hlr.LowRankHinv.get_diagonal(correction_pointwise_variance)
        post_pointwise_variance = pr_pointwise_variance - correction_pointwise_variance
        return post_pointwise_variance, pr_pointwise_variance, correction_pointwise_variance
        
        
