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
        
class GaussianLRPosteriorMixture:

    """
    Class that implements a mixture of GaussianLRPosterior objects.
    """

    def __init__(self, prior, components=None, mix_weights=None):

        """
        Constructor.

        Inputs:

        1. :code:`prior` The prior distribution from which these posteriors \
                         are derived. All components are assumed to be derived \
                         from the same prior.

        2. :code:`components` Initial components in the mixture. Must be a \
                              list of GaussianLRPosterior objects if supplied.

        3. :code:`mix_weights`  If supplied, must be numpy array of weights of \
                                each component in the mixture. If mix_weights 
                                doesn't sum to one, it'll be normalized.

        """

        self.prior = prior

        if components:

            self.components = components[:]
            self.ncomp      = len(self.components)

            if mix_weights is None:
                raise TypeError("Mixture weights cannot be NoneType if components are supplied")
            else:
                self.mix_weights = mix_weights / np.sum(mix_weights)

            self.set_log_det_prod()

        else:
            self.components  = []
            self.mix_weights = None
            self.ncomp       = 0



    def set_log_det_prod(self):
        """

        Save the log of the determinants of the product of prior 
        covariance and the full hessian for each component, 
        :math:`\\log|R^{-1} (R + H_{misfit}[i])| = \\log | I + R^{-1}
        H_{misfit}[i] |` 

        """

        #c.d       := eigenvalues of R^{-1} H_misfit
        #c.d + 1.0 := eigenvalues of I + R^{-1} H_misfit
        self.log_det_prod = np.fromiter(((np.sum(np.log(c.d + 1.0))) \
                                        for c in self.components), \
                                        dtype = np.float64)


    def getISRatio(self, m):

        """
        Returns the importance sampling weight :math:`\\frac{prior(m)}{mixture(m)}`

        Inputs: 

        1. :code:`m`: :code:`dolfin.Vector` at which IS weight needs to be computed

        Outputs:

        1. :math:`\\frac{prior(m)}{mixture(m)}`
        """

        norm_prior = self.prior.R.inner(m - self.prior.mean, \
                                        m - self.prior.mean)

        norm_post  = np.fromiter((c.Hlr.inner(m - c.mean, m - c.mean) \
                                 for c in self.components), \
                                 dtype = np.float64)

        return 1.0 / np.sum(np.exp(0.5 * (norm_prior - norm_post \
                                          + self.log_det_prod) \
                                   + np.log(self.mix_weights)))

    def init_vector(self, x, dim):
        """
        Initialize a vector :code:`x` to be compatible with the range/domain of
        the covariance of each component or random white noise.
        
        Inputs:

        1. :code:`x`   : :code:`dolfin.Vector` to be initialized

        2. :code:`dim` : :code:`0` - range, :code:`1` - domain, :code:`"noise"` - white noise

        """

        self.components[0].init_vector(x, dim)

    def sample(self, idx, *args):
        """
        Sample from the :code:`idx` th component
        
        Possible calls:
        
        1) :code:`sample(idx, s_prior, s_mix)`

           Given a *zero mean* prior sample, obtain a mixture sample.

           Inputs:

           - :code:`idx`     : Index of the mixture component to sample from 

           - :code:`s_prior` : :code:`dolfin.Vector` that holds prior sample

           - :code:`s_mix`   : :code:`dolfin.Vector` that will hold mixture sample

        2) :code:`sample(noise, s_prior, s_mix)`

           Given a :code:`noise` vector, sample :code:`s_prior` from the prior
           and :code:`s_mix` from the mixture.

           Inputs:
           
           - :code:`idx`     : Index of the mixture component to sample from 

           - :code:`noise`   : :code:`dolfin.Vector` containing white noise, :math:`\\sim N(0, I)`

           - :code:`s_prior` : :code:`dolfin.Vector` that will hold prior sample

           - :code:`s_mix`   : :code:`dolfin.Vector` that will hold mixture sample
        
        Output:
            None
        """

       
        #To actually sample, use the underlying GaussianLRPosterior object's
        #sampler
        if len(args) == 2:
            self.components[idx].sample(args[0], args[1])
        elif len(args) == 3:
            self.components[idx].sample(args[0], args[1], args[2])
        else:
            raise NameError('Invalid number of parameters in GaussianLRPosteriorMixture::sample')
       
    def append(self, posterior, new_mix_weights):
        """

        Add a new component to the Gaussian mixture

        Inputs:

        - :code:`posterior`       : A new :code:`GaussianLRPosterior` object to add to the mixture

        - :code:`new_mix_weights` : :code:`numpy` array of new mixture weights

        Outputs:
            None
        """

        self.components.append(posterior)
        self.ncomp += 1
        self.mix_weights = new_mix_weights / np.sum(new_mix_weights)
        self.set_log_det_prod()
