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
from dolfin                import Vector, Function, MPI
from scipy.stats           import norm
from ...modeling.variables import *
from ...modeling.misfit    import QOIMisfit
from ...utils.random       import parRandom

class MCSampler:
    """
    Class for performing a rare event Monte Carlo simulation
    """


    def __init__(self, target, model, qoi):

        """
        Constructor

        Inputs:

        - :code:`target` - An instance of the :code:`rare_events.Target` that describes the limits of the target interval

        - :code:`model` - A `hippylib.model` instance that describes the \
                           governing PDE and the prior uncertainty.

        - :code:`qoi` - A `hippylib.forward_uq.qoi` instance that describes \
                         the quantity of interest
            
        """
        self.target = target
        self.qoi    = qoi
        self.model  = model

        self.n_samples      = 1
        self.dump_to_file   = False
        self.load_from_file = False
        self.filename       = None
        
        noise_std_dev = self.target.limits.max() - self.target.limits.min()
        self.model.misfit.noise_variance = noise_std_dev * noise_std_dev

    def rnd(self):
        """
        Generate a sample from the prior
        """
        
        noise = Vector()
        self.model.prior.init_vector(noise, "noise")
        prior_sample = self.model.problem.generate_parameter()
        parRandom.normal(1.0, noise)
        
        self.model.prior.sample(noise, prior_sample)

        return prior_sample

    def run(self):
        """
        Run the Monte Carlo sampler. 

        Outputs: 

        - :code:`avg`        : The rare event probability estimate

        - :code:`RMSE / avg` : Relative RMSE error in the probability estimate 

        - :code:`frac`       : the fraction of samples that end up inside the target

        """

        rank = MPI.rank(self.model.prior.R.mpi_comm())
        
        if self.load_from_file == True:
            samples = np.loadtxt(self.filename)
            weights = samples[:, self.model.problem.Vh[PARAMETER].dim() + 1]
        else:
            if self.dump_to_file == True:
                f = open(self.filename, 'wb')
                np.set_printoptions(threshold=np.nan)
                dim = self.model.problem.Vh[PARAMETER].dim()
                np.set_printoptions(edgeitems=dim + 2)
                np.set_printoptions(linewidth=np.inf)
        
            if rank == 0:
                print("Generating ", self.n_samples, "MC samples\n")
            
            weights = np.zeros(self.n_samples)
            
            for i in (range(self.n_samples)):
                sample = self.rnd()
                [u, _, p] = self.model.generate_vector()
                self.model.solveFwd(u, [u, sample, p])

                obs = self.qoi.eval([u, sample, p])

                if obs < self.target.limits[1] and obs > self.target.limits[0]:
                    weights[i] = 1.0
                    
                if self.dump_to_file == True:
                    output = np.r_[sample.get_local().T, obs, weights[i]]
                    output_str = str(output)
                    output_str_without_bracket = output_str[2:-2]
                    f.write(b'%s\n' % output_str_without_bracket)
                    f.flush()
              
        avg  = np.mean(weights, dtype=np.float64)
        RMSE = np.sqrt(np.var(weights) / self.n_samples)
        frac = np.count_nonzero(weights) / float(self.n_samples)
        
        return (avg, RMSE / avg, frac)
