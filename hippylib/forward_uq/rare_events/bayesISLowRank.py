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
from dolfin                     import Vector, MPI
from .bimcOptProblem            import BIMCOptProblem
from .linearApprox              import LinearApproximator
from .optLikeVar                import ParamOptimizer
from ...utils.random            import parRandom
from ...modeling.variables      import *
from ...modeling.model          import Model
from ...modeling.posterior      import GaussianLRPosterior, \
                                       GaussianLRPosteriorMixture

   
class Target(object):
    """
    Target class with the following attributes:

    - :code:`loc` - location of the measurement/observation in the spatial \
                     domain. Optional, defaults to None if \
                     observation/measurement is not a  variable of spatial coordinates

    - :code:`limits` - Start and end points of the target interval :math:`\\mathbb{Y}`

    - :code:`grid` - Array containing elements that will be inverted for. 
    
    - :code:`width` - Measure of :math:`\\mathbb{Y}`
    """
    
    def __init__(self, limits, grid, loc=None):
        self.loc = loc
        self.limits = limits
        self.grid = grid
        self.width = self.limits.max() - self.limits.min()


class BayesianImpSamplerLowRank(object):
    """
    Class that implements the Bayesian Inverse Monte Carlo algorithm.
    """
    
    def __init__(self, opt_problem, limits, init_noise_var):
        """
        Constructor

        Inputs:
          
        - :code:`opt_problem` - A :code:`BIMCOptProblem` instance 
        - :code:`limits` - A list of limits of the target interval in \
                            ascending order, :math:`[y_{\min}, y_{\max}]`
        - :code:`init_noise_var` - Initial guess noise variance. Used for \
                                    computing optimal sampling parameters in \
                                    :code:`BayesianImpSamplerLowRank.initialize()`
        
        """
        
        self.opt_problem    = opt_problem
        self.limits         = limits
        self.init_noise_var = init_noise_var
        
        
        self.mixture = GaussianLRPosteriorMixture(self.opt_problem.model.prior)
        
        self.comm = self.mixture.prior.R.mpi_comm()

        self.mixture_exist = False
        
        
        #Default parameters. May be updated by the user as required.
        self.n_samples = 100 # number of samples to estimate the rare
                                     # event probability
        
        #Initial guess for the NewtonCG solver
        self.solver_init_guess = self.opt_problem.model.prior.mean.copy()

        
    def initialize(self, nopt=None):
        """
        Initialize the importance sampler by computing the optimal \
        pseudo likelihood variance (noise variance) and the optimal \
        pseudo data point. 
        
        If :code:`nopt` is specified, :code:`nopt` evenly spaced \
        points in the target interval are used as the data instead. \
        Each pseudo data point contributes to a component in the final \
        IS mixture. 

        Inputs:
        
        - :code:`nopt` : Number of components in the IS mixture
        """

        rank = MPI.rank(self.comm)

        if rank == 0:
            print("Initializing ...")

        #Find map corresponding to mid point of the target interval

        self.opt_problem.model.misfit.noise_variance = self.init_noise_var
        
        mid     = 0.5 * (self.limits[0] + self.limits[1])
        mid_sol = self.opt_problem.getMAP(mid, self.solver_init_guess)
        qoi_mid = self.opt_problem.getQOIFromState(mid_sol)

        # Assert that mid_sol is meaningful, i.e., it lies inside the target
        assert qoi_mid < self.limits[1] and qoi_mid > self.limits[0]

        #Build linear approximation around mid MAP to obtain optimal parameters
        linear_approx = LinearApproximator(mid_sol[PARAMETER], self.opt_problem)
        linear_approx.build()

        #Obtain optimal parameters
        opt_param       = ParamOptimizer(self.limits, linear_approx)
        param           = opt_param.getOptParam()
        noise_variance  = param[0]
        data            = param[1]
        
        self.opt_problem.model.misfit.noise_variance = noise_variance

        if nopt is None or nopt == 1:
            target_arr = np.array([data])
        else:
            target_arr = np.linspace(self.limits[0], self.limits[1], nopt)

        self.target_obj = Target(np.array(self.limits), target_arr)

        if rank == 0:
            print("Done.")

        self.initialized = True

    
    def createNormals(self):
        """
        Create array of GaussianLRPosterior instances.         
        """

        if not self.initialized:
            self.initialize()

        if not self.mixture_exist:
            self._createNormalsFromScratch()
    
    
    def _createNormalsFromScratch(self):
        """
        This routine creates the GaussianLRPosteriorMixture. Each component of
        the mixture corresponds to the Gaussian approximation of the posterior
        when data is :code:`target_obj.grid[i]`.

        This is done by finding the MAP point corresponding to each data point 
        and creating a Hessian object. Only a low rank approximation of the hessian
        is used to construct the Gaussian. The low rank approximations is 
        constructed by solving a generalized eigenvalue problem.         

        """
        
        nopt = self.target_obj.grid.shape[0]

        for i in (range(nopt)):

            map_pt = self.opt_problem.getMAP(self.target_obj.grid[i],
                                             self.solver_init_guess.copy())
            
            #Obtain generalized eigenvalues of GN approximation of misfit
            #hessian
            eig_val, eig_vec = self.opt_problem.getHessLR(map_pt)
            
            #Discard negative eigenvalues
            eig_val = np.maximum(eig_val, np.zeros(eig_val.shape))

            #Construct a posterior approximation and add it to the IS mixture
            #ones(i + 1) represent new unnormalized IS weights
            self.mixture.append(GaussianLRPosterior(self.opt_problem.model.prior, \
                                                    eig_val, eig_vec),
                                                    np.ones(i + 1)) 
            self.mixture.components[i].mean = map_pt[PARAMETER].copy()

        self.mixture_exist = True

    def _getPriorSample(self, add_mean=True):
        """
        Returns a sample from the prior
        
        Inputs: 
        
        - :code:`prior`   - :code:`hippylib._Prior` instance
        - :code:`add_mean` - :code:`bool`, should the prior mean be added to the sample?
        
        Outputs
    
        - :code:`prior_sample` - :code:`dolfin.Vector` that holds the sample from :code:`prior`
    
        """
    
        noise = Vector()
        self.opt_problem.model.prior.init_vector(noise, "noise")
        parRandom.normal(1.0, noise);
        
        prior_sample = Vector()
        self.opt_problem.model.prior.init_vector(prior_sample, 0) 
        
        self.opt_problem.model.prior.sample(noise, prior_sample, add_mean)
    
        return prior_sample

    def rnd(self):
    
        """
        Returns a sample from the importance mixture

        Output:

        - :code:`posterior_sample` : :code:`dolfin.Vector` instance \
                                      containing a mixture sample
        """
       
        assert self.mixture_exist

        rank = MPI.rank(self.comm)
        
        # First a prior sample is generated. 
        #prior sample needs to be centered around 0
        prior_sample = self._getPriorSample(add_mean=False)

        #Then, this prior sample is converted to a posterior sample 
        posterior_sample = self.opt_problem.model.problem.generate_parameter()
            
        #Ensure that the same component in the mixture is being sampled from
        if rank == 0:
            idx = np.random.randint(0, self.mixture.ncomp)
        else:
            idx = None

        idx = self.comm.tompi4py().bcast(idx, root=0)
            
        self.mixture.sample(idx, prior_sample, posterior_sample)
        
        return posterior_sample

    def run(self, print_samples=False, samples_file_name=None):
        """
        Runs the importance sampler to compute the rare event probability.
        
        Inputs:

        - :code:`print_samples`     : Save the generated samples and weights

        - :code:`samples_file_name` : File where the samples and weights will be saved

        Outputs: 

        - :code:`avg`        : The rare event probability estimate

        - :code:`RMSE / avg` : Relative RMSE error in the probability estimate 

        - :code:`ESS`        : Effective Sample Size
        
        - :code:`frac`       : the fraction of samples that end up inside the target

        """

        if not self.initialized:
            self.initialize()
        
        if not self.mixture_exist:
            self.createNormals()


        if print_samples: 
            if samples_file_name is not None:
                f = open(samples_file_name, 'w')
                np.set_printoptions(threshold=np.nan)
                dim = self.opt_problem.model.problem.Vh[PARAMETER].dim()
                np.set_printoptions(edgeitems=(dim + 3))
                np.set_printoptions(linewidth=np.inf)
            else:
                raise ValueError
        
        indicator_weights = np.zeros(self.n_samples)
        weights           = np.zeros(self.n_samples)

        rank = MPI.rank(self.comm)

        if rank == 0:
            print("Generating %d samples" % self.n_samples)

        
        for i in (range(self.n_samples)):
           
            m_imp_sample = self.rnd()
            
            obs = self.opt_problem.getQOI(m_imp_sample)

            weights[i] = self.mixture.getISRatio(m_imp_sample)

            if obs < self.limits[1] and obs > self.limits[0]: 
                indicator_weights[i] = weights[i]
 
            if print_samples:
                output = np.r_[m_imp_sample.get_local().T, obs,\
                                  indicator_weights[i], weights[i]]
                output_str = str(output)
                output_str_without_bracket = output_str[2:-1]
                f.write(b'%s\n' % output_str_without_bracket)
                f.flush()
               
        avg  = np.mean(indicator_weights, dtype=np.float64)
        
        RMSE = np.sqrt(np.var(indicator_weights, dtype=np.float64) / self.n_samples)
       
        weight_normalized = indicator_weights / np.sum(indicator_weights)
        
        ESS = 1.0 / (np.sum(weight_normalized ** 2.0) * self.n_samples)
        
        frac = np.count_nonzero(indicator_weights) / float(self.n_samples)
        
        non_zero_indicator_weights = indicator_weights[indicator_weights > 0]
        
        dKL = np.sum(non_zero_indicator_weights * np.log(non_zero_indicator_weights)) \
                / (avg * self.n_samples) - np.log(avg)

        return (avg, RMSE / avg, ESS, dKL, frac)

    def verifyInversion(self):
        """
        Returns the push forward of the MAP points

        Outputs:

        - List of QOIs corresponding to each component mean
        """
        
        return [self.opt_problem.getQOI(c.mean) \
                for c in (self.mixture)]
              
