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

import unittest
from mpi4py import MPI
from petsc4py import PETSc
import dolfin as dl


import hippylib as hp

class TestSNES(unittest.TestCase):
    
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(MPI.COMM_WORLD, 12, 15)
        self.V = dl.FunctionSpace(self.mesh, "Lagrange", 1)
        self.u = dl.Function(self.V)
        self.v = dl.TestFunction(self.V)
        self.F = dl.inner(5.0, self.v) * dl.dx - dl.sqrt(self.u * self.u) * dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx - dl.inner(self.u, self.v) * dl.dx
        
        self.bc = dl.DirichletBC(self.V, dl.Constant(1.), "on_boundary")

    def test_set_petsc_options(self):
        """Test setting PETSc options for SNES solver"""
        opts = {"ksp_type": "minres"}  # something other than the default
        optmgr = hp.OptionsManager(opts, "test")
        
        problem = hp.SNES_VariationalProblem(self.F, self.u, [self.bc])
        solver = hp.SNES_VariationalSolver(problem, MPI.COMM_WORLD, optmgr)
        
        assert solver.snes.getKSP().getType() == "minres"
        assert solver.snes.getOptionsPrefix() == "test_"

    def test_snes_variational_problem(self):
            """Test Newton Problem for a simple nonlinear PDE
            
            FEniCS 2019.1.0 version of the DolfinX example:
            https://github.com/FEniCS/dolfinx/blob/b6864c032e5e282f9b73f80523f8c264d0c7b3e5/python/test/unit/nls/test_newton.py#L190
            """            
            problem = hp.SNES_VariationalProblem(self.F, self.u, [self.bc])
            self.u.assign(dl.Constant(0.9))  # initial guess
            
            b_vec = dl.PETScVector()
            J_mat = dl.PETScMatrix()
            
            snes = PETSc.SNES().create()
            snes.setFunction(problem.evalFunction, b_vec.vec())
            snes.setJacobian(problem.evalJacobian, J_mat.mat())
            
            snes.setTolerances(rtol=1.0e-9, max_it=10)
            snes.getKSP().setType("preonly")
            snes.getKSP().setTolerances(rtol=1.0e-9)
            snes.getKSP().getPC().setType("lu")

            snes.solve(None, problem.u.vector().vec())
            assert snes.getConvergedReason() > 0
            assert snes.getIterationNumber() < 6
            
            # Modify boundary condition and solve again
            bc = dl.DirichletBC(self.V, dl.Constant(0.6), "on_boundary")
            problem = hp.SNES_VariationalProblem(self.F, self.u, [self.bc])
            
            snes.solve(None, problem.u.vector().vec())
            assert snes.getConvergedReason() > 0
            assert snes.getIterationNumber() < 6

            snes.destroy()
            
            
    def test_snes_variational_solver(self):
        """Test Newton Problem/Solver for a simple nonlinear PDE
        
        FEniCS 2019.1.0 version of the DolfinX example:
        https://github.com/FEniCS/dolfinx/blob/b6864c032e5e282f9b73f80523f8c264d0c7b3e5/python/test/unit/nls/test_newton.py#L190
        """
        opts = {"ksp_type": "preonly", "ksp_rtol": 1.0e-9, "pc_type": "lu"}
        optmgr = hp.OptionsManager(opts, "solver")
        
        problem = hp.SNES_VariationalProblem(self.F, self.u, [self.bc])
        self.u.assign(dl.Constant(0.9))  # initial guess
        solver = hp.SNES_VariationalSolver(problem, MPI.COMM_WORLD, optmgr)
        its, reason = solver.solve()
        
        assert its > 0
        assert solver.snes.getConvergedReason() > 0
            