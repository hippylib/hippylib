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
# Software Foundation) version 2.0 dated June 1991.

import math
import argparse
import dolfin as dl
import ufl
import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
import numpy as np

try:
    import matplotlib.pyplot as plt
    has_plt = True
except:
    has_plt = False

class GammaCenter(dl.SubDomain):
    def inside(self, x, on_boundary):
        return ( abs(x[1]-.5) < dl.DOLFIN_EPS )

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
    parser = argparse.ArgumentParser(description='Model Subsurface')
    parser.add_argument('--nx',
                        default=32,
                        type=int,
                        help="Number of elements in x-direction")
    parser.add_argument('--ny',
                        default=64,
                        type=int,
                        help="Number of elements in y-direction")
    parser.add_argument('--nsamples',
                        default=500,
                        type=int,
                        help="Number of MC samples")
    args = parser.parse_args()
    dl.set_log_active(False)
    sep = "\n"+"#"*80+"\n"
    ndim = 2
    nx = args.nx
    ny = args.ny
    
    dl.parameters["ghost_mode"] = "shared_facet"
    mesh = dl.UnitSquareMesh(nx, ny)
    
    rank = dl.MPI.rank(mesh.mpi_comm())
        
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print( sep, "Set up the mesh and finite element spaces", sep)
        print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs) )
    
    # Initialize Expressions
    f = dl.Constant(0.0)
        
    u_bdr = dl.Expression("x[1]", degree=1)
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
     
    gamma = .1
    delta = .5
    
    theta0 = 2.
    theta1 = .5
    alpha  = math.pi/4
    
    anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(theta0, theta1, alpha)
    mtrue = true_model(Vh[PARAMETER], gamma, delta,anis_diff)

    locations = np.array([[0.1, 0.1], [0.1, 0.9], [.5,.5], [.9, .1], [.9, .9]])

    pen = 1e1
    prior = MollifiedBiLaplacianPrior(Vh[PARAMETER], gamma, delta, locations, mtrue, anis_diff, pen)
    
    if rank == 0:    
        print( "Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2) )
        
    GC = GammaCenter()
    marker = dl.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    marker.set_all(0)
    GC.mark(marker, 1)
    dss = dl.Measure("dS", domain=mesh, subdomain_data=marker)
    n = dl.Constant((0.,1.))#dl.FacetNormal(Vh[STATE].mesh())

    def qoi_varf(u,m):
        return ufl.avg(ufl.exp(m)*ufl.dot( ufl.grad(u), n) )*dss(1)

    qoi = VariationalQoi(Vh,qoi_varf) 
    p2qoimap = Parameter2QoiMap(pde, qoi)
    
    if True:
        parameter2QoiMapVerify(p2qoimap, prior.mean, eps=np.power(.5, np.arange(20,0,-1)), plotting = True )
    
    k = 100    
    Omega = MultiVector(prior.mean, k)
    parRandom.normal(1., Omega)
    
    q_taylor = TaylorApproximationQoi(p2qoimap, prior)
    q_taylor.computeLowRankFactorization(Omega)
    
    if rank == 0:
        plotEigenvalues(q_taylor.d)
    
    e_lin  = q_taylor.expectedValue(order=1)
    e_quad = q_taylor.expectedValue(order=2)
    v_lin  = q_taylor.variance(order=1)
    v_quad = q_taylor.variance(order=1)
    if rank == 0:
        print( "E[Q_lin] = {0:7e}, E[Q_quad] = {1:7e}".format(e_lin, e_quad))
        print( "Var[Q_lin] = {0:7e}, Var[Q_quad] = {1:7e}".format(v_lin, v_quad) )
    
    varianceReductionMC(prior, p2qoimap, q_taylor,  nsamples=args.nsamples)
    
    if rank == 0 and has_plt:
        plt.show()
