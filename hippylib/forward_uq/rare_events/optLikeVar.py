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
from dolfin          import MPI
from scipy.special   import erf
from scipy.special   import erfc
from scipy.special   import erfcx
from scipy.stats     import norm
from scipy.optimize  import minimize
from .linearApprox   import *
from .bimcOptProblem import NotConvergedException

def getTruncStats(avg, variance, d_min, d_max):
    """

    Function to get 0th, 1st, and 2nd moments of a truncated normal
    distribution. The Normal PDF is assumed to be univariate and
    truncated between :code:`d_min` and :code:`d_max`. Modified to avoid
    underflow. See https://github.com/cossio/TruncatedNormal.jl
    
    Inputs:

        - :code:`avg`: mean of original Normal distribution
        - :code:`variance`: variance of original Normal distribution
        - :code:`d_min`, :code:`d_max`: Truncation limits
    
    Outputs:

        - :code:`trunc_mu`, :code:`truncAvg`, :code:`trunc_var` - the 0th, 1st \
                    and 2nd moments of the truncated Normal respectively.
                 
    """
    
    sqrt2  = np.sqrt(2)
    sqrtpi = np.sqrt(np.pi)
    
    std_dev = np.sqrt(variance)
    
    alpha = (d_min - avg) / std_dev
    beta  = (d_max - avg) / std_dev
    
    alpha_scaled = alpha / sqrt2
    beta_scaled  = beta / sqrt2
    
    trunc_mu = 0.5 * _erfdiff(alpha_scaled, beta_scaled)
    
    trunc_mean = avg + sqrt2 / sqrtpi \
                    * std_dev * _trunc_helper1(alpha_scaled, beta_scaled);
    
    trunc_var = variance * (1.0 \
                + 2.0 / sqrtpi * _trunc_helper2(alpha_scaled, beta_scaled) \
                - 2.0 / np.pi * (_trunc_helper1(alpha_scaled, beta_scaled)) ** 2.0);
    
    return trunc_mu, trunc_mean, trunc_var

def _erfdiff(x, y):
    """
    Returns :code:`erf(y)` - :code:`erf(x)`. Modified to avoid underflow
    """
    
    if abs(x) > abs(y):

        diff = - _erfdiff(y, x)
    
    elif abs(x - y) < 1e-7:
        
        #compute diff using taylor expansion
        diff = _erfdiff_taylor(x, y - x)

    elif min(x, y) >= 0.0:
        
        err = np.exp(x * x - y * y)
        
        diff = np.exp(-x * x) * ( erfcx(x) - err * erfcx(y) )
    
    elif max(x, y) < 0.0:

        err = np.exp(x * x - y * y)

        diff = np.exp(-x * x) * (err * erfcx(-y) - erfcx(-x))
    else:
        diff = erf(y) - erf(x)

    return diff

def _erfdiff_taylor(x, delta):
    """
    Returns :code:`erf(x + delta)` - :code:`erf(x)`, where delta is a small number
    """

    x2 = x * x;

    c1 = 1.0 - x2 * (1.0 + 0.5 * x2);
    c2 = - x * (1.0 + x2);
    c3 = - 1.0 / 3.0 + x2;
    c4 = 0.5 * x;
    c5 = 0.1;

    diff = 2.0 / np.sqrt(np.pi) * (c1 + (c2 + (c3 \
               + (c4 + c5 * delta) * delta) * delta) * delta) * delta;

    return diff


def _trunc_helper1(x, y):
    """
    Helper function for getTruncStats
    """

    diff = np.exp(x * x - y * y);

    if abs(x) > abs(y):

        val = _trunc_helper1(y, x)

    elif abs(x - y) < 1e-7:

        val = _trunc_helper1_taylor(x, y - x);

    elif min(x, y) > 0.0:

        val = (1 - diff) / (erfcx(x) - diff * erfcx(y));

    elif max(x, y) < 0.0:

        val = (1 - diff) / (diff * erfcx(-y) - erfcx(-x));

    else:

        val = np.exp(-x * x) * (1 - diff) / (erf(y) - erf(x));

    return val


def _trunc_helper2(x, y):
    """
    Helper function for getTruncStats
    """

    diff = np.exp(x * x - y * y)

    if abs(x) > abs(y):
        val = _trunc_helper2(y, x)
    elif abs(x - y) < 1e-7:
        val = _trunc_helper2_taylor(x, y - x)
    elif min(x, y) > 0.0:
        val = (x - diff * y) / (erfcx(x) - diff * erfcx(y))
    elif max(x, y) < 0.0:
        val = (x - diff * y) / (diff * erfcx(-y) - erfcx(-x))
    else:
        val = np.exp(-x * x) * (x - diff * y) / (erf(y) - erf(x))

    return val


def _trunc_helper1_taylor(x, delta):
    """
    Helper function for getTruncStats
    """

    sqrtpi = np.sqrt(np.pi);

    c0 = sqrtpi * x;
    c1 = sqrtpi / 2.0;
    c2 = - 1.0 / 6.0 * sqrtpi * x;
    c3 = - 1.0 / 12.0 * sqrtpi;
    c4 = 1.0 / 90.0 * sqrtpi * x * (x * x + 1.0);

    diff = c0 + (c1 + (c2 + (c3 + c4 * delta) * delta) * delta) * delta;    

    return diff

def _trunc_helper2_taylor(x, delta):
    """
    Helper function for getTruncStats
    """

    sqrtpi = np.sqrt(np.pi)

    x2 = x * x

    c0 = 0.5 * sqrtpi * (2.0 * x2 - 1.0)
    c1 = sqrtpi * x
    c2 = - 1.0 / 3.0 * sqrtpi * (x2 - 1.0)
    c3 = - 1.0 / 3.0 * sqrtpi * x
    c4 = 1.0 / 90.0 * sqrtpi * ((2.0 * x2 + 3.0) * x2 - 8.0)

    diff = c0 + (c1 + (c2 + (c3 + c4 * delta) * delta) * delta) * delta

    return diff

class ParamOptimizer:
    """
    Class to obtain the optimal likelihood variance and optimal data point 
    to perform BIMC with.
    """
    
    def __init__(self, interval, linear_approx):
        """
        Constructor 

        Arguments:

            - :code:`interval`: interval over which probability is required

            - :code:`linear_approximator` - :code:`hippylib.forward_uq.rare_events.LinearApproximator` \
                                    instance that contains the linear approximation
        """

        self.linear_approx = linear_approx
        self.interval = interval

        
        #prior push forward mean and variance
        self.push_fwd_var = linear_approx.lin_std_dev ** 2.
        self.push_fwd_mean = self.linear_approx.eval(self.linear_approx.prior.mean)
        
    def getKLDist(self, param):
        """
        Returns KL dist between the current IS distribution and the ideal one
        """

        var = param[0]
        delta = param[1]

        d_hat = self.push_fwd_mean
        gamma_hat_sq = self.push_fwd_var

        alpha_sq = var / (var + gamma_hat_sq)
        
        (mu, d_tilde, gamma_tilde_sq) = getTruncStats(d_hat, gamma_hat_sq, 
                                                   self.interval[0], 
                                                   self.interval[1])

        dKL = (np.log(np.sqrt(alpha_sq) / mu) 
              + ((delta - d_tilde) ** 2.0 + gamma_tilde_sq) / (2.0 * var) 
              - ((delta - d_hat) ** 2.0) / (2.0 * (var + gamma_hat_sq)))

        return dKL

    def getOptParam(self, bounds=None, tol=1e-6, tolX=1e-8):
        """

        Minimizes the KL divergence w.r.t. the pseudo-likelihood variance and
        the pseudo-data point and returns the minimizers.

        """
        rank = MPI.rank(self.linear_approx.prior.R.mpi_comm())

        if rank == 0:
            print("Obtaining optimal parameters ...\n")

        init_guess = np.array(
                        [self.linear_approx.opt_problem.model.misfit.noise_variance,
                         0.5 * (self.interval[0] + self.interval[1])])


        if bounds is None:
            res = minimize(self.getKLDist, init_guess, method='Nelder-Mead', tol=tol)
                           #options={'xtol':tolX})
        else:
            res = minimize(self.getKLDist, init_guess, 
                           bounds=bounds, method='L-BFGS-B')#, method='bounded', options={'xatol':tolX})

        if not res.success:
            raise NotConvergedException("Failed to obtain optimal parameters\n Exit code: %d\nExit message: %s" % (res.status, res.message))

        return res.x


            

