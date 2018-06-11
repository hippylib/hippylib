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

import dolfin as dl
import math
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *


def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def true_model(Vh, gamma, delta, anis_diff):
    prior = BiLaplacianPrior(Vh, gamma, delta, anis_diff )
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise,mtrue)
    return mtrue
            
if __name__ == "__main__":
    dl.set_log_active(False)
    sep = "\n"+"#"*80+"\n"
    ndim = 2
    nx = 64
    ny = 64
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
        return dl.exp(m)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx
    
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    if dlversion() <= (1,6,0):
        pde.solver = dl.PETScKrylovSolver("cg", amg_method())
        pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", amg_method())
        pde.solver_adj_inc = dl.PETScKrylovSolver("cg", amg_method())
    else:
        pde.solver = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
        pde.solver_fwd_inc = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
        pde.solver_adj_inc = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver.parameters["relative_tolerance"] = 1e-15
    pde.solver.parameters["absolute_tolerance"] = 1e-20
    pde.solver_fwd_inc.parameters = pde.solver.parameters
    pde.solver_adj_inc.parameters = pde.solver.parameters
 
    ntargets = 300
    np.random.seed(seed=1)
    targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    if rank == 0:
        print ("Number of observation points: {0}".format(ntargets) )
    misfit = PointwiseStateObservation(Vh[STATE], targets)
    
    gamma = .1
    delta = .5
    
    anis_diff = dl.Expression(code_AnisTensor2D, degree = 1)
    anis_diff.theta0 = 2.
    anis_diff.theta1 = .5
    anis_diff.alpha = math.pi/4
    mtrue = true_model(Vh[PARAMETER], gamma, delta,anis_diff)
        
    locations = np.array([[0.1, 0.1], [0.1, 0.9], [.5,.5], [.9, .1], [.9, .9]])

    pen = 1e1
    prior = MollifiedBiLaplacianPrior(Vh[PARAMETER], gamma, delta, locations, mtrue, anis_diff, pen)
    
    if rank == 0:
        print ( "Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2) )   
                
    #Generate synthetic observations
    utrue = pde.generate_state()
    x = [utrue, mtrue, None]
    pde.solveFwd(x[STATE], x, 1e-9)
    misfit.B.mult(x[STATE], misfit.d)
    rel_noise = 0.01
    MAX = misfit.d.norm("linf")
    noise_std_dev = rel_noise * MAX
    parRandom.normal_perturb(noise_std_dev, misfit.d)
    misfit.noise_variance = noise_std_dev*noise_std_dev
    
    model = Model(pde,prior, misfit)
    
    if rank == 0:
        print( sep, "Test the gradient and the Hessian of the model", sep )
    
    m0 = dl.interpolate(dl.Expression("sin(x[0])", element=Vh[PARAMETER].ufl_element() ), Vh[PARAMETER])
    modelVerify(model, m0.vector(), 1e-12, is_quadratic = False, verbose = (rank == 0) )

    if rank == 0:
        print( sep, "Find the MAP point", sep)
    m = prior.mean.copy()
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-9
    parameters["abs_tolerance"] = 1e-12
    parameters["max_iter"]      = 25
    parameters["inner_rel_tolerance"] = 1e-15
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
    Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], misfit_only=True)
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
    
    fid = dl.File("results/pointwise_variance.pvd")
    fid << vector2Function(post_pw_variance,Vh[PARAMETER], name="Posterior")
    fid << vector2Function(pr_pw_variance, Vh[PARAMETER], name="Prior")
    fid << vector2Function(corr_pw_variance, Vh[PARAMETER], name="Correction")

    if rank == 0:
        print (sep, "Save State, Parameter, Adjoint, and observation in paraview", sep)
    xxname = ["State", "Parameter", "Adjoint"]
    xx = [vector2Function(x[i], Vh[i], name=xxname[i]) for i in range(len(Vh))]
    dl.File("results/poisson_state.pvd") << xx[STATE]
    dl.File("results/poisson_state_true.pvd") << vector2Function(utrue, Vh[STATE], name = xxname[STATE])
    dl.File("results/poisson_parameter.pvd") << xx[PARAMETER]
    dl.File("results/poisson_parameter_true.pvd") << vector2Function(mtrue, Vh[PARAMETER], name = xxname[PARAMETER])
    dl.File("results/poisson_parameter_prmean.pvd") << vector2Function(prior.mean, Vh[PARAMETER], name = xxname[PARAMETER])
    dl.File("results/poisson_adjoint.pvd") << xx[ADJOINT]
        
    exportPointwiseObservation(Vh[STATE], misfit.B, misfit.d, "results/poisson_observation")
    
    if rank == 0:
        print( sep, "Generate samples from Prior and Posterior\n","Export generalized Eigenpairs", sep )
    fid_prior = dl.File("samples/sample_prior.pvd")
    fid_post  = dl.File("samples/sample_post.pvd")
    nsamples = 500
    noise = dl.Vector()
    posterior.init_vector(noise,"noise")
    s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
    s_post = dl.Function(Vh[PARAMETER], name="sample_post")
    for i in range(nsamples):
        parRandom.normal(1., noise)
        posterior.sample(noise, s_prior.vector(), s_post.vector())
        fid_prior << s_prior
        fid_post << s_post
        
    #Save eigenvalues for printing:
    
    U.export(Vh[PARAMETER], "hmisfit/evect.pvd", varname = "gen_evects", normalize = True)
    if rank == 0:
        np.savetxt("hmisfit/eigevalues.dat", d)
    
    if nproc == 1:
        print( sep, "Visualize results", sep)
        dl.plot(xx[STATE], title = xxname[STATE])
        dl.plot(dl.exp(xx[PARAMETER]), title = xxname[PARAMETER])
        dl.plot(xx[ADJOINT], title = xxname[ADJOINT])
        dl.interactive()
    
    if rank == 0:
        plt.figure()
        plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
        plt.yscale('log')
        plt.show()    
        
