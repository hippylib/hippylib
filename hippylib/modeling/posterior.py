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

from dolfin import Vector, Function, File
from ..algorithms.lowRankOperator import LowRankOperator
import numpy as np

class LowRankHessian:
    """
    Operator that represents the action of the low rank approximation
    of the Hessian and of its inverse.
    """
    def __init__(self, prior, d, U):
        self.prior = prior
        self.LowRankH = LowRankOperator(d, U)
        dsolve = d / (np.ones(d.shape, dtype=d.dtype) + d)
        self.LowRankHinv = LowRankOperator(dsolve, U)
        self.help = Vector(U[0].mpi_comm())
        self.init_vector(self.help, 0)
        self.help1 = Vector(U[0].mpi_comm())
        self.init_vector(self.help1, 0)
        
    def init_vector(self,x, dim):
        self.prior.init_vector(x,dim)
    
    def inner(self,x,y):
        Hx = Vector(self.help.mpi_comm())
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
    Object to sample from the low rank approximation
    of the posterior.
    
        .. math:: y = ( I - U S U^{T}) x,
    
    where

    :math:`S = I - (I + D)^{-1/2}, x \\sim \\mathcal{N}(0, R^{-1}).`
    """
    def __init__(self, prior, d, U):
        self.prior = prior        
        ones = np.ones( d.shape, dtype=d.dtype )        
        self.d = ones - np.power(ones + d, -.5)
        self.lrsqrt = LowRankOperator(self.d, U)
        self.help = Vector(U[0].mpi_comm())
        self.init_vector(self.help, 0)
        
    def init_vector(self,x, dim):
        self.prior.init_vector(x,dim)
        
    def sample(self, noise, s):
        self.prior.R.mult(noise, self.help)
        self.lrsqrt.mult(self.help, s)
        s.axpy(-1., noise)
        s *= -1.

class GaussianLRPosterior:
    """
    Class for the low rank Gaussian Approximation of the Posterior.
    This class provides functionality for approximate Hessian
    apply, solve, and Gaussian sampling based on the low rank
    factorization of the Hessian.
    
    In particular if :math:`d` and :math:`U` are the dominant eigenpairs of
    :math:`H_{\\mbox{misfit}} U[:,i] = d[i] R U[:,i]`
    then we have:
    
    - low rank Hessian apply: :math:`y = ( R + RU D U^{T}) x.`
    - low rank Hessian solve: :math:`y = (R^-1 - U (I + D^{-1})^{-1} U^T) x.`
    - low rank Hessian Gaussian sampling: :math:`y = ( I - U S U^{T}) x`, where :math:`S = I - (I + D)^{-1/2}` and :math:`x \\sim \\mathcal{N}(0, R^{-1}).`
    """
    def __init__(self, prior, d, U, mean=None):
        """
        Construct the Gaussian approximation of the posterior.
        Input:
        - :code:`prior`: the prior mode.
        - :code:`d`:     the dominant generalized eigenvalues of the Hessian misfit.
        - :code:`U`:     the dominant generalized eigenvector of the Hessian misfit :math:`U^T R U = I.`
        - :code:`mean`:  the MAP point.
        """
        self.prior = prior
        self.d = d
        self.U = U
        self.Hlr = LowRankHessian(prior, d, U)
        self.sampler = LowRankPosteriorSampler(self.prior, self.d, self.U)
        self.mean=None
        
        
        
    def cost(self, m):
        if self.mean is None:
            return .5*self.Hlr.inner(m,m)
        else:
            dm = m - self.mean
            return .5*self.Hlr.inner(dm,dm)
            
        
    def init_vector(self,x, dim):
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`H`.
        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        self.prior.init_vector(x,dim)
        
    def sample(self, *args, **kwargs):
        """
        possible calls:
        
        1) :code:`sample(s_prior, s_post, add_mean=True)`

           Given a prior sample  :code:`s_prior` compute a sample :code:`s_post` from the posterior.

           - :code:`s_prior` is a sample from the prior centered at 0 (input).
           - :code:`s_post` is a sample from the posterior (output).
           - if :code:`add_mean=True` (default) then the samples will be centered at the map point.
             
        2) :code:`sample(noise, s_prior, s_post, add_mean=True)`
        
           Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s_prior` from the prior and 
           :code:`s_post` from the posterior.

           - :code:`noise` is a realization of white noise (input).
           - :code:`s_prior` is a sample from the prior (output).
           - :code:`s_post`  is a sample from the posterior.
           - if :code:`add_mean=True` (default) then the prior and posterior samples will be centered at the respective means.
        
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
    
    def trace(self, **kwargs):
        """
        Compute/estimate the trace of the posterior, prior distribution
        and the trace of the data informed correction.
        
        See :code:`_Prior.trace` for more details.
        """
        pr_trace = self.prior.trace(**kwargs)
        corr_trace = self.trace_update()
        post_trace = pr_trace - corr_trace
        return post_trace, pr_trace, corr_trace
    
    def trace_update(self):
        return self.Hlr.LowRankHinv.trace(self.prior.M)
    
    def pointwise_variance(self, **kwargs):
        """
        Compute/estimate the pointwise variance of the posterior, prior distribution
        and the pointwise variance reduction informed by the data.
        
        See :code:`_Prior.pointwise_variance` for more details. 
        """
        pr_pointwise_variance = self.prior.pointwise_variance(**kwargs)
        correction_pointwise_variance = Vector(self.prior.R.mpi_comm())
        self.init_vector(correction_pointwise_variance, 0)
        self.Hlr.LowRankHinv.get_diagonal(correction_pointwise_variance)
        post_pointwise_variance = pr_pointwise_variance - correction_pointwise_variance
        return post_pointwise_variance, pr_pointwise_variance, correction_pointwise_variance
    
    def klDistanceFromPrior(self, sub_comp = False):
        dplus1 = self.d + np.ones_like(self.d)
        
        c_logdet = 0.5*np.sum( np.log(dplus1) )
        c_trace  = -0.5*np.sum(self.d/dplus1)
        c_shift  = self.prior.cost(self.mean)
        
        kld = c_logdet + c_trace + c_shift
        
        if sub_comp:
            return kld, c_logdet, c_trace, c_shift
        else:
            return kld
        
        
