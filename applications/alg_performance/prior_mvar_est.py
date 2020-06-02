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

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Marginal Variance Estimation')
    parser.add_argument('--nx',
                        default=64,
                        type=int,
                        help="Number of elements in x-direction")
    parser.add_argument('--ny',
                        default=64,
                        type=int,
                        help="Number of elements in y-direction")
    parser.add_argument('--nt',
                        default=10,
                        type=int,
                        help="Number of estimates")
    parser.add_argument('--p',
                        default=5,
                        type=int,
                        help="Number of eigenvalues to drop")

    args = parser.parse_args()
    try:
        dl.set_log_active(False)
    except:
        pass
    sep = "\n"+"#"*80+"\n"
    ndim = 2
    nx = args.nx
    ny = args.ny
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
    
    nvs = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 2560], dtype=np.int)
    
    err_est  = np.zeros((nvs.shape[0], args.nt), dtype=np.float64)
    err_rand = np.zeros((nvs.shape[0], args.nt), dtype=np.float64)
    
    for i in np.arange(nvs.shape[0]):
        nv = nvs[i]
        for t in range(args.nt):
            pr_pw_variance_1 = prior.pointwise_variance(method="Estimator", k=2*nv)
            pr_pw_variance_1.axpy(-1., pr_pw_variance_exact)
            err = pr_pw_variance_1.norm("l2")
            err_est[i, t] = err
            
    for i in np.arange(nvs.shape[0]):
        nv = nvs[i]
        for t in range(args.nt):
            pr_pw_variance_2 = prior.pointwise_variance(method="Randomized", k=nv-args.p, p = args.p)
            pr_pw_variance_2.axpy(-1., pr_pw_variance_exact)
            err = pr_pw_variance_2.norm("l2")
            err_rand[i, t] = err
            
    if rank == 0:
        print(nvs)
        print(np.mean(err_est, axis=1))
        print(np.mean(err_rand, axis=1))
        
    np.savetxt('err_est.txt', err_est)
    np.savetxt('err_rand.txt', err_rand)
    