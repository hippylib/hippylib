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

import dolfin as dl
import numpy as np
from ..utils.random import parRandom

def varianceReductionMC(prior, rqoi, taylor_qoi, nsamples, filename="realizations.txt"):
    """
    This function computes Monte Carlo Estimates for forward propagation of uncertainty.
    The uncertain parameter satisfies a Gaussian distribution with known mean and covariance 
    (describes as the inverse of a differential operator).
    Convergence of the Monte Carlo estimates is accelerated using a variance reduction
    techinque based on a Taylor approximation of the parameter-to-qoi map.
    
    Inputs:

        - :code:`prior` - an object of type :code:hIPPYlib._Prior` that allows to generate samples from the prior distribution
        - :code:`rqoi` - an object of type :code:`ReducedQOI` that describes the parameter-to-qoi map
        - :code:`taylor_qoi` - an object of type :code:`TaylorApproximationQOI` that computes the first and second order Taylor \
        approximation of the qoi
        - :code:`nsamples` - an integer representing the number of samples for the MC estimates
        - :code:`filename` - a string containing the name of the file where the computed qoi \
        and its Taylor approximations (for each realization of the parameter) are saved
                    
    Outputs:

        - Sample mean of the quantity of interest `q`, its Taylor approx `q1` and `q2`, and corrections `y1=q-q1` and `y2=q-q2`.
        - MSE (Mean square error) of the standard MC, and the variance reduced MC using `q1` and `q2`.
    
    .. note:: The variate control MC estimator can be computed off-line by postprocessing the file containing the \
    values of the computed qoi and Taylor approximations.
    
    """
    noise = dl.Vector(prior.R.mpi_comm())
    sample = dl.Vector(prior.R.mpi_comm())
    
    prior.init_vector(noise, "noise")
    prior.init_vector(sample, 1)
    
    rank = dl.MPI.rank(noise.mpi_comm())
    
    q_i = np.zeros(nsamples)
    q1_i = np.zeros(nsamples)
    q2_i = np.zeros(nsamples)
    y1_i = np.zeros(nsamples)
    y2_i = np.zeros(nsamples)
    
    Eq1_exact = taylor_qoi.expectedValue(order=1)
    Eq2_exact = taylor_qoi.expectedValue(order=2)
    
    if rank == 0:
        print( "nsamples | E[q], E[y1] + E[q1],  E[y2] + E[q2]| Var[q] Var[y1] Var[y2]")
        fid = open(filename,"w")
    
    for i in range(nsamples):
        parRandom.normal(1., noise)
        prior.sample(noise, sample)
        q_i[i] = rqoi.reduced_eval(sample)
        q1_i[i] = taylor_qoi.eval(sample, order=1)
        q2_i[i] = taylor_qoi.eval(sample, order=2)
        y1_i[i] = q_i[i] - q1_i[i]
        y2_i[i] = q_i[i] - q2_i[i]
        
        if rank == 0:
            fid.write("{0:15e} {1:15e} {2:15e}\n".format(q_i[i], q1_i[i], q2_i[i]))
            fid.flush()
        
        if ( (i+1) % 10 == 0) or (i+1 == nsamples):
            Eq  = np.sum(q_i)/float(i+1)
            Eq1 = np.sum(q1_i)/float(i+1)
            Eq2 = np.sum(q2_i)/float(i+1)
            Ey1 = np.sum(y1_i)/float(i+1)
            Ey2 = np.sum(y2_i)/float(i+1)
            
            Varq = np.sum(np.power(q_i,2))/float(i) - (float(i+1)/float(i)*Eq*Eq)
            Varq1 = np.sum(np.power(q1_i,2))/float(i) - (float(i+1)/float(i)*Eq1*Eq1)
            Varq2 = np.sum(np.power(q2_i,2))/float(i) - (float(i+1)/float(i)*Eq2*Eq2)
            Vary1 = np.sum(np.power(y1_i,2))/float(i) - (float(i+1)/float(i)*Ey1*Ey1)
            Vary2 = np.sum(np.power(y2_i,2))/float(i) - (float(i+1)/float(i)*Ey2*Ey2)
            
            if rank == 0:
                print( "{0:3} | {1:7e} {2:7e} {3:7e} | {4:7e} {5:7e} {6:7e}".format(
                    i+1, Eq, Ey1+Eq1_exact, Ey2+Eq2_exact,
                         Varq, Vary1, Vary2))
                
    Vq1_exact = taylor_qoi.variance(order=1) 
    Vq2_exact = taylor_qoi.variance(order=1)         
    
    if rank == 0:        
        fid.close()
            
        print( "Expected value q1: analytical: ", Eq1_exact, "estimated: ", Eq1)
        print( "Expected value q2: analytical: ", Eq2_exact, "estimated: ", Eq2)
        print( "Variance q1: analytical", Vq1_exact, "estimated: ", Varq1)
        print( "Variance q2: analytical", Vq2_exact, "estimated: ", Varq2)
    
    return Eq, Ey1, Ey2, Eq1_exact, Eq2_exact, Varq/nsamples, Vary1/nsamples, Vary2/nsamples
