# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2022, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
# Copyright (c) 2023-2024, The University of Texas at Austin 
# & University of California--Merced.
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

import argparse
import numpy as np
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl
import dolfin as dl
import matplotlib.pyplot as plt


from hippylib import *

# Helper functions and classes.

def u_boundary(x, on_boundary):
    """Boundary marker for the top and bottom walls (y-direction).
    """
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    """Boundary marker for left and right walls (x-direction).
    """
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

class HeatEquationVarf:
    """Variational form for the heat equation.
    """
    def __init__(self, dt, f):
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.f = f
            
    @property
    def dt(self):
        return self._dt
        
    def __call__(self,u,u_old, m, p, t):
        return (u - u_old)*p*self.dt_inv*ufl.dx + ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - self.f*p*ufl.dx


def true_model(prior):
    """Return a noisy sample from the prior.
    """
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise,mtrue)
    return mtrue


# Main script.
def main(args):
    # Fixed parameters.
    SEP = "\n"+"#"*80+"\n"
    NDIM = 2
    T_INIT = 0.
    T_FINAL = 1.
    GAMMA = 0.1
    DELTA = 0.5
    
    # Unpack command line arguments.
    nx = args.nx
    ny = args.ny
    
    # Set up mesh.
    mesh = dl.UnitSquareMesh(nx, ny)
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())
    
    # Set up variational spaces for state and parameter.
    Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh = [Vh2, Vh1, Vh2]
    
    # Report the number of DOFs.
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print(SEP, "Set up the mesh and finite element spaces", SEP, flush=True)
        print(f"Number of dofs: state = {ndofs[STATE]}, parameter = {ndofs[PARAMETER]}, adjoint = {ndofs[ADJOINT]}", flush=True)
    
    # Initialize expressions for forcing, boundary condition, initial condition.
    f = dl.Constant(0.0)
    u_bdr = dl.Expression("0.0", element=Vh[STATE].ufl_element())
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    u0_expr = dl.Expression("x[0]*(1.-x[0])*x[1]*(1.-x[1])", element=Vh[STATE].ufl_element())
    u0 = dl.interpolate(u0_expr, Vh[STATE])
    
    # Set up time stepping and variational problem.
    dt = (T_FINAL - T_INIT)/args.nt
    pde_varf = HeatEquationVarf(dt, f)
    pde = TimeDependentPDEVariationalProblem(Vh, pde_varf, bc, bc0, u0, T_INIT, T_FINAL, is_fwd_linear=True)
    
    # Set up prior on parameter.
    prior = BiLaplacianPrior(Vh[PARAMETER], GAMMA, DELTA, robin_bc=True)
    mtrue = true_model(prior)    
    if rank == 0:
        print(f"Prior regularization: (delta_x - gamma*Laplacian)^order: delta={DELTA}, gamma={GAMMA}, order=2", flush=True)
    
    # Generate synthetic observations.
    utrue = pde.generate_state()
    x = [utrue, mtrue, None]
    pde.solveFwd(x[STATE], x)
    
    rel_noise = 0.01
    max_state = x[STATE].norm("linf", "linf")
    noise_std_dev = rel_noise * max_state
    
    # Set up misfit object.
    misfits = []
    for t in pde.times:
        misfit_t = ContinuousStateObservation(Vh[STATE], ufl.dx, bc0)
        misfit_t.d.axpy(1., x[STATE].view(t))
        parRandom.normal_perturb(noise_std_dev, misfit_t.d)
        misfit_t.noise_variance = noise_std_dev*noise_std_dev
        misfits.append(misfit_t)
    
    misfit = MisfitTD(misfits, pde.times)
    
    # Set up inverse problem.
    model = Model(pde, prior, misfit)
    if rank == 0:
        print(SEP, "Test the gradient and the Hessian of the model", SEP, flush=True)
    
    m0 = dl.interpolate(dl.Expression("sin(x[0])", element=Vh[PARAMETER].ufl_element()), Vh[PARAMETER])
    modelVerify(model, m0.vector(), is_quadratic=False, misfit_only=True, verbose=(rank==0))
    
    if rank == 0:
        print(SEP, "Find the MAP point", SEP, flush=True)
    m = model.prior.mean.copy()
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-9
    parameters["abs_tolerance"] = 1e-12
    parameters["max_iter"]      = 25
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 5
    if rank != 0:
        parameters["print_level"] = -1
    else:
        parameters.showMe()
    
    solver = ReducedSpaceNewtonCG(model, parameters)
    
    # Solve the inverse problem.
    x = solver.solve([None, m, None])
    if rank == 0:
        if solver.converged:
            print(f"\nConverged in {solver.it} iterations.", flush=True)
        else:
            print("\nNot Converged", flush=True)
        
        print(f"Termination reason: {solver.termination_reasons[solver.reason]}", flush=True)
        print(f"Final gradient norm: {solver.final_grad_norm}", flush=True)
        print(f"Final cost: {solver.final_cost}", flush=True)
        
    if rank == 0:
        print(SEP, "Compute the low rank Gaussian Approximation of the posterior", SEP, flush=True)
    
    # Compute Laplace approximation.
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    Hmisfit = ReducedHessian(model, misfit_only=True)
    k = 50
    p = 20
    if rank == 0:
        print(f"Double Pass Algorithm. Requested eigenvectors: {k}, oversampling {p}.", flush=True)
    
    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)
    
    d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
    posterior = GaussianLRPosterior(prior, d, U)
    posterior.mean = x[PARAMETER]
    post_tr, prior_tr, corr_tr = posterior.trace(method="Randomized", r=200)
    if rank == 0:
        print(f"Posterior trace: {post_tr}; Prior trace: {prior_tr}; Correction trace: {corr_tr}", flush=True)
        
    post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance(method="Randomized", r=200)
    
    # Report Results.
    kl_dist = posterior.klDistanceFromPrior()
    if rank == 0:
        print(f"KL distance: {kl_dist}", flush=True)
    
    with dl.XDMFFile(mesh.mpi_comm(), "results/heat/pointwise_variance.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        fid.write(vector2Function(post_pw_variance, Vh[PARAMETER], name="Posterior"), 0)
        fid.write(vector2Function(pr_pw_variance, Vh[PARAMETER], name="Prior"), 0)
        fid.write(vector2Function(corr_pw_variance, Vh[PARAMETER], name="Correction"), 0)
        
    if rank == 0:
        print(SEP, "Saving parameter", SEP, flush=True)
    
    with dl.XDMFFile(mesh.mpi_comm(), "results/heat/results.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        fid.write(vector2Function(x[PARAMETER], Vh[PARAMETER], name = "map"), 0)
        fid.write(vector2Function(mtrue, Vh[PARAMETER], name = "true parameter"), 0)
        fid.write(vector2Function(prior.mean, Vh[PARAMETER], name = "prior mean"), 0)
    
    pde.exportState(x[STATE], "results/heat/state.xdmf")
    
    if rank == 0:
        print(SEP, "Generate samples from Prior and Posterior\n", "Export generalized Eigenpairs", SEP, flush=True)
    
    nsamples = args.nsamples
    noise = dl.Vector()
    posterior.init_vector(noise, "noise")
    s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
    s_post = dl.Function(Vh[PARAMETER], name="sample_post")
    with dl.XDMFFile(mesh.mpi_comm(), "results/heat/samples.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        for i in range(nsamples):
            parRandom.normal(1., noise)
            posterior.sample(noise, s_prior.vector(), s_post.vector())
            fid.write(s_prior, i)
            fid.write(s_post, i)
    
    # Save eigenvalues for printing.
    U.export(Vh[PARAMETER], "results/heat/evec.xdmf", varname="gen_evec", normalize=True)
    if rank == 0:
        np.savetxt("results/heat/eigenvalues.txt", d)
    
    if rank == 0:
        plt.figure()
        plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
        plt.yscale('log')
        plt.show()


# CLI argument parsing.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heat equation model.")
    parser.add_argument('--nx',
                        default=32,
                        type=int,
                        help="Number of elements in x-direction")
    parser.add_argument('--ny',
                        default=32,
                        type=int,
                        help="Number of elements in y-direction")
    parser.add_argument('--nt',
                        default=32,
                        type=int,
                        help="Number of time steps")
    parser.add_argument('--nsamples',
                        default=50,
                        type=int,
                        help="Number of samples from prior and Laplace Approximation")
    args = parser.parse_args()
    
    try:
        dl.set_log_active(False)
    except:
        pass
    
    main(args)