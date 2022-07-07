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


def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def true_model(prior):
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise,mtrue)
    return mtrue
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Subsurface')
    parser.add_argument('--nx',
                        default=64,
                        type=int,
                        help="Number of elements in x-direction")
    parser.add_argument('--ny',
                        default=64,
                        type=int,
                        help="Number of elements in y-direction")
    parser.add_argument('--nsamples',
                        default=50,
                        type=int,
                        help="Number of samples from prior and Laplace Approximation")
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
            
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print (sep, "Set up the mesh and finite element spaces", sep)
        print ("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs) )
    
    # Initialize Expressions
    f = dl.Constant(0.0)
        
    u_bdr = dl.Expression("x[1]", element = Vh[STATE].ufl_element() )
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    
    def pde_varf(u,m,p):
        return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx
    
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    pde.solver = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver_fwd_inc = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver_adj_inc = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
       
    pde.solver.parameters["relative_tolerance"] = 1e-15
    pde.solver.parameters["absolute_tolerance"] = 1e-20
    pde.solver_fwd_inc.parameters = pde.solver.parameters
    pde.solver_adj_inc.parameters = pde.solver.parameters
 
    ntargets = 50
    np.random.seed(seed=1)
    #Targets only on the bottom
    targets_x = np.random.uniform(0.1,0.9, [ntargets] )
    targets_y = np.random.uniform(0.1,0.5, [ntargets] )
    targets = np.zeros([ntargets, ndim])
    targets[:,0] = targets_x
    targets[:,1] = targets_y
    #targets everywhere
    #targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    if rank == 0:
        print ("Number of observation points: {0}".format(ntargets) )
    Mpar = 1e4
    misfit = MultPointwiseStateObservation(Vh[STATE], targets, Mpar)
    
    gamma = .1
    delta = .5
    
    theta0 = 2.
    theta1 = .5
    alpha  = math.pi/4
    
    anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(theta0, theta1, alpha)
    
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True )
    
    mtrue = true_model(prior)
            
    if rank == 0:
        print ( "Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2) )   
                
    #Generate synthetic observations
    utrue = pde.generate_state()
    x = [utrue, mtrue, None]
    pde.solveFwd(x[STATE], x)
    misfit.B.mult(x[STATE], misfit.d)
    print(misfit.d.get_local())
    parRandom.speckle(Mpar, misfit.d)
    print(misfit.d.get_local())

    
    

    
    model = Model(pde,prior, misfit)
    
    if rank == 0:
        print( sep, "Test the gradient and the Hessian of the model", sep )
    
    m0 = dl.interpolate(dl.Expression("sin(x[0])", element=Vh[PARAMETER].ufl_element() ), Vh[PARAMETER])
    modelVerify(model, m0.vector(), is_quadratic = False, misfit_only=True, verbose = (rank == 0) )

    if rank == 0:
        print( sep, "Find the MAP point", sep)
    m = prior.mean.copy()
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-9
    parameters["abs_tolerance"] = 1e-12
    parameters["max_iter"]      = 25
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 5
    if rank != 0:
        parameters["print_level"] = -1
        
    if rank == 0:
        parameters.showMe()
    solver = ReducedSpaceNewtonCG(model, parameters)
    
    x = solver.solve([None, m, None])
    
    if rank == 0:
        if solver.converged:
            print( "\nConverged in ", solver.it, " iterations.")
        else:
            print( "\nNot Converged")

        print ("Termination reason: ", solver.termination_reasons[solver.reason])
        print ("Final gradient norm: ", solver.final_grad_norm)
        print ("Final cost: ", solver.final_cost)
        
    if rank == 0:
        print (sep, "Compute the low rank Gaussian Approximation of the posterior", sep)
    
    model.setPointForHessianEvaluations(x, gauss_newton_approx = False)
    Hmisfit = ReducedHessian(model, misfit_only=True)
    k = 50
    p = 20
    if rank == 0:
        print ("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )
    
    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)

    d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
    posterior = GaussianLRPosterior(prior, d, U)
    posterior.mean = x[PARAMETER]
    
    
    post_tr, prior_tr, corr_tr = posterior.trace(method="Randomized", r=200)
    if rank == 0:
        print ("Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}".format(post_tr, prior_tr, corr_tr))

    post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance(method="Randomized", r=200)
        
    kl_dist = posterior.klDistanceFromPrior()
    if rank == 0:
        print ("KL-Distance from prior: ", kl_dist)
    
    with dl.XDMFFile(mesh.mpi_comm(), "results/pointwise_variance.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
    
        fid.write(vector2Function(post_pw_variance, Vh[PARAMETER], name="Posterior"), 0)
        fid.write(vector2Function(pr_pw_variance, Vh[PARAMETER], name="Prior"), 0)
        fid.write(vector2Function(corr_pw_variance, Vh[PARAMETER], name="Correction"), 0)

    if rank == 0:
        print (sep, "Save State, Parameter, Adjoint, and observation in paraview", sep)
    xxname = ["state", "parameter", "adjoint"]
    xx = [vector2Function(x[i], Vh[i], name=xxname[i]) for i in range(len(Vh))]
    
    with dl.XDMFFile(mesh.mpi_comm(), "results/results.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False 
           
        fid.write(xx[STATE],0)
        fid.write(vector2Function(utrue, Vh[STATE], name = "true state"), 0)
        fid.write(xx[PARAMETER],0)
        fid.write(vector2Function(mtrue, Vh[PARAMETER], name = "true parameter"), 0)
        fid.write(vector2Function(prior.mean, Vh[PARAMETER], name = "prior mean"), 0)
        fid.write(xx[ADJOINT],0)
        
    exportPointwiseObservation(Vh[STATE], misfit.B, misfit.d, "results/poisson_observation")
    
    if rank == 0:
        print( sep, "Generate samples from Prior and Posterior\n","Export generalized Eigenpairs", sep )

    nsamples = args.nsamples
    noise = dl.Vector()
    posterior.init_vector(noise,"noise")
    s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
    s_post = dl.Function(Vh[PARAMETER], name="sample_post")
    with dl.XDMFFile(mesh.mpi_comm(), "results/samples.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        for i in range(nsamples):
            parRandom.normal(1., noise)
            posterior.sample(noise, s_prior.vector(), s_post.vector())
            fid.write(s_prior, i)
            fid.write(s_post, i)
        
    #Save eigenvalues for printing:
    
    U.export(Vh[PARAMETER], "results/evect.xdmf", varname = "gen_evects", normalize = True)
    if rank == 0:
        np.savetxt("results/eigevalues.dat", d)
        
    if rank == 0:
        plt.figure()
        plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
        plt.yscale('log')
        plt.show()    
        
