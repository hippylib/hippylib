# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
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
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *

            
def run(nx, ny, nvs):
    ndim = 2
    mesh = dl.UnitSquareMesh(nx, ny)
    
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())
            
    Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    
    ndofs = Vh.dim()
    
    if rank == 0:
        print ("Number of dofs: {0}".format(ndofs) )

    gamma = .1
    delta = .5
    
    theta0 = 2.
    theta1 = .5
    alpha  = math.pi/4
    
    anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(theta0, theta1, alpha)
    
    prior = BiLaplacianPrior(Vh, gamma, delta, anis_diff, robin_bc=True )
    prior.Asolver = PETScLUSolver(mesh.mpi_comm(), "mumps")
    prior.Asolver.set_operator(prior.A)
    
    pr_pw_variance_exact = prior.pointwise_variance(method="Exact")
    
    norm = pr_pw_variance_exact.norm("l2")
    print(norm)
        
    data  = np.zeros((nvs.shape[0],), dtype=np.float64)
         
    for i in np.arange(nvs.shape[0]):
        nv = nvs[i]
        pr_pw_variance_2 = prior.pointwise_variance(method="Randomized", r=nv)
        pr_pw_variance_2.axpy(-1., pr_pw_variance_exact)
        err = pr_pw_variance_2.norm("l2")
        data[i] = err/norm
        
    return data, ndofs
            

    
if __name__ == "__main__":
    try:
        dl.set_log_active(False)
    except:
        pass
    sep = "\n"+"#"*80+"\n"
    
    nvs = np.array([32, 64, 128], dtype=np.int)
    nx  = np.array([32, 64, 128, 256], dtype=np.int)
    
    data = np.zeros((nx.shape[0], nvs.shape[0]+1), dtype=np.float64)
    
    for i in np.arange(nx.shape[0]):
        
        out, ndofs = run(nx[i], nx[i], nvs)
        data[i,0] = ndofs
        data[i, 1:] = out[:]
        
    
    np.savetxt('data_cov_estimation_scaling.txt', data, header='ndofs err_32 err_64 err_128')
    
    plt.loglog(data[:,0], data[:,1], '-b', label='bar{r} = 32')
    plt.loglog(data[:,0], data[:,2], '-r', label='bar{r} = 64')
    plt.loglog(data[:,0], data[:,3], '-g', label='bar{r} = 128')
    plt.legend()
    plt.show()
    