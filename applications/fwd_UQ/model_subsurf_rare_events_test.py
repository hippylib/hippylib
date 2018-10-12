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

class PoissonOptProblem(BIMCOptProblem):

    def __init__(self, model):

        self.model = model
        self.solver = ReducedSpaceNewtonCG(self.model)

        self.solver.parameters["rel_tolerance"] = 1e-5
        self.solver.parameters["abs_tolerance"] = 1e-10

class PointwiseQOI:

    """
    Implements a QOI that represents the value of the state at some target
    location
    """

    def __init__(self, V, target_pt):

        """
        Constructor:

        V :        Function space of the state variable
        target_pt : Spatial location at which the state variable is being
                    measured
        """

        
        v        = dl.TestFunction(V)
        dummy    = dl.Constant('0.0') * v * dl.dx
        self.chi = dl.assemble(dummy)
        
        delta    = dl.PointSource(V, dl.Point(target_pt[0], target_pt[1]), 1.0)
        delta.apply(self.chi)

        self.state = dl.Function(V).vector()

    def eval(self, x):

        return self.chi.inner(x[STATE])

    def grad(self, i, x, g):

        if i == STATE:

            self.grad_state(x, g)

        elif i == PARAMETER:
            g.zero()

        else:
            raise i

    def grad_state(self, x, g):

        g.zero()
        g.axpy(1.0, self.chi)
         
    def apply_ij(self, i, j, dir, out):
         
        out.zero()
    
    def setLinearizationPoint(self, x):

        self.state.zero()
        self.state.axpy(1.0, x[STATE])


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

    #This script demonstrates the use of the BayesianImpSamplerLowRank class for
    #computing rare event probabilities.

    #Specify the forward problem and its adjoints
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
     
    #Specify the distribution that describes the uncertainty in the parameter.
    #The uncertain parameter is the log transmissibility field.
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
        
    #Let the QOI be the pressure at (0.1, 0.5)
    targetLoc = np.array([0.1, 0.5])
    qoi = PointwiseQOI(Vh[STATE], targetLoc)
    misfit = QOIMisfit(qoi, pde)
    
    model = Model(pde, prior, misfit)
    opt_problem = PoissonOptProblem(model)
    
    n_samples = 1000
    nopt = 10
    
    #Specify limits of the target interval
    limits = [0.4239, 0.4974]
    
    #Specify an initial guess for the pseudo-likelihood noise variance
    init_noise_var = 0.01 * (limits[1] - limits[0]) ** 2
    
    imp_sampler = BayesianImpSamplerLowRank(opt_problem, limits, init_noise_var)
    imp_sampler.initialize(nopt = nopt)
    imp_sampler.createNormals()
    
    target = Target(np.array(limits), np.array([limits[0]]))
    mc_sampler = MCSampler(target, model, qoi)
    
    n_samples_list = [10, 100, 1000, 10000]
    
    mc_file = open("mc_results.txt", "aw")
    is_file = open("is_results.txt", "aw")

    for n in n_samples_list:

        mc_sampler.n_samples = n
        mc_results           = imp_sampler.run()

        imp_sampler.n_samples = n
        is_results            = imp_sampler.run()

        np.savetxt(mc_file, [n, mc_results[0], mc_results[1]])
        np.savetxt(is_file, [n, is_results[0], is_results[1]])

        print(n)
        print(mc_results)
        print(is_results)
    
    mc_file.close()
    is_file.close()
