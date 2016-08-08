# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import dolfin as dl
import math
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append( "../../" )
from hippylib import *


def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def true_model(Vh, gamma, delta, anis_diff):
    prior = BiLaplacianPrior(Vh, gamma, delta, anis_diff )
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    noise_size = noise.array().shape[0]
    noise.set_local( np.random.randn( noise_size ) )
    atrue = dl.Vector()
    prior.init_vector(atrue, 0)
    prior.sample(noise,atrue)
    return atrue
            
if __name__ == "__main__":
    dl.set_log_active(False)
    sep = "\n"+"#"*80+"\n"
    print sep, "Set up the mesh and finite element spaces", sep
    ndim = 2
    nx = 64
    ny = 64
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    print "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim())
    
    # Initialize Expressions
    f = dl.Expression("0.0")
        
    u_bdr = dl.Expression("x[1]")
    u_bdr0 = dl.Expression("0.0")
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    
    def pde_varf(u,a,p):
        return dl.exp(a)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx
    
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0)
 
    ntargets = 300
    np.random.seed(seed=1)
    targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    print "Number of observation points: {0}".format(ntargets)
    misfit = PointwiseStateObservation(Vh[STATE], targets)
    
    
    gamma = .1
    delta = .5
    
    anis_diff = dl.Expression(code_AnisTensor2D)
    anis_diff.theta0 = 2.
    anis_diff.theta1 = .5
    anis_diff.alpha = math.pi/4
    atrue = true_model(Vh[PARAMETER], gamma, delta,anis_diff)
        
    locations = np.array([[0.1, 0.1], [0.1, 0.9], [.5,.5], [.9, .1], [.9, .9]])

    pen = 1e1
    prior = MollifiedBiLaplacianPrior(Vh[PARAMETER], gamma, delta, locations, atrue, anis_diff, pen)
        
    print "Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2)    
                
    #Generate synthetic observations
    utrue = pde.generate_state()
    x = [utrue, atrue, None]
    pde.solveFwd(x[STATE], x, 1e-9)
    misfit.B.mult(x[STATE], misfit.d)
    rel_noise = 0.01
    MAX = misfit.d.norm("linf")
    noise_std_dev = rel_noise * MAX
    randn_perturb(misfit.d, noise_std_dev)
    misfit.noise_variance = noise_std_dev*noise_std_dev
    
    model = Model(pde,prior, misfit)
           
    print sep, "Test the gradient and the Hessian of the model", sep
    a0 = dl.interpolate(dl.Expression("sin(x[0])"), Vh[PARAMETER])
    modelVerify(model, a0.vector(), 1e-12)

    print sep, "Find the MAP point", sep
    a0 = prior.mean.copy()
    solver = ReducedSpaceNewtonCG(model)
    solver.parameters["rel_tolerance"] = 1e-9
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"]      = 25
    solver.parameters["inner_rel_tolerance"] = 1e-15
    solver.parameters["c_armijo"] = 1e-4
    solver.parameters["GN_iter"] = 5
    
    x = solver.solve(a0)
    
    if solver.converged:
        print "\nConverged in ", solver.it, " iterations."
    else:
        print "\nNot Converged"

    print "Termination reason: ", solver.termination_reasons[solver.reason]
    print "Final gradient norm: ", solver.final_grad_norm
    print "Final cost: ", solver.final_cost
        
    print sep, "Compute the low rank Gaussian Approximation of the posterior", sep
    model.setPointForHessianEvaluations(x)
    Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], gauss_newton_approx=False, misfit_only=True)
    k = 50
    p = 20
    print "Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p)
    Omega = np.random.randn(x[PARAMETER].array().shape[0], k+p)
    #d, U = singlePassG(Hmisfit, model.R, model.Rsolver, Omega, k, check_Bortho=True, check_Aortho=True, check_residual=True)
    d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, check_Bortho=False, check_Aortho=False, check_residual=False)
    posterior = GaussianLRPosterior(prior, d, U)
    posterior.mean = x[PARAMETER]
    
    post_tr, prior_tr, corr_tr = posterior.trace(method="Estimator", tol=1e-1, min_iter=20, max_iter=100)
    print "Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}".format(post_tr, prior_tr, corr_tr)
    post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance("Exact")

    print sep, "Save State, Parameter, Adjoint, and observation in paraview", sep
    xxname = ["State", "Parameter", "Adjoint"]
    xx = [dl.Function(Vh[i], x[i], name=xxname[i]) for i in range(len(Vh))]
    dl.File("results/poisson_state.pvd") << xx[STATE]
    dl.File("results/poisson_state_true.pvd") << dl.Function(Vh[STATE], utrue, name = xxname[STATE])
    dl.File("results/poisson_parameter.pvd") << xx[PARAMETER]
    dl.File("results/poisson_parameter_true.pvd") << dl.Function(Vh[PARAMETER], atrue, name = xxname[PARAMETER])
    dl.File("results/poisson_parameter_prmean.pvd") << dl.Function(Vh[PARAMETER], prior.mean, name = xxname[PARAMETER])
    dl.File("results/poisson_adjoint.pvd") << xx[ADJOINT]
        
    exportPointwiseObservation(targets, misfit.d, "results/poisson_observation.vtp")
    
    fid = dl.File("results/pointwise_variance.pvd")
    fid << dl.Function(Vh[PARAMETER], post_pw_variance, name="Posterior")
    fid << dl.Function(Vh[PARAMETER], pr_pw_variance, name="Prior")
    fid << dl.Function(Vh[PARAMETER], corr_pw_variance, name="Correction")
    
    
    print sep, "Generate samples from Prior and Posterior\n","Export generalized Eigenpairs", sep
    fid_prior = dl.File("samples/sample_prior.pvd")
    fid_post  = dl.File("samples/sample_post.pvd")
    nsamples = 500
    noise = dl.Vector()
    posterior.init_vector(noise,"noise")
    noise_size = noise.array().shape[0]
    s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
    s_post = dl.Function(Vh[PARAMETER], name="sample_post")
    for i in range(nsamples):
        noise.set_local( np.random.randn( noise_size ) )
        posterior.sample(noise, s_prior.vector(), s_post.vector())
        fid_prior << s_prior
        fid_post << s_post
        
    #Save eigenvalues for printing:
    posterior.exportU(Vh[PARAMETER], "hmisfit/evect.pvd")
    np.savetxt("hmisfit/eigevalues.dat", d)
    
    print sep, "Visualize results", sep
    dl.plot(xx[STATE], title = xxname[STATE])
    dl.plot(dl.exp(xx[PARAMETER]), title = xxname[PARAMETER])
    dl.plot(xx[ADJOINT], title = xxname[ADJOINT])
    
    plt.figure()
    plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
    plt.yscale('log')
        
    plt.show()    
    dl.interactive()
    
