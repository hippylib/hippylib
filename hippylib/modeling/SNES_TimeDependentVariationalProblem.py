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

import time
import dolfin as dl
import ufl
from .TimeDependentPDEVariationalProblem import TimeDependentPDEVariationalProblem
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms import SNES_VariationalProblem, SNES_VariationalSolver
from ..utils.vector2function import vector2Function
from ..utils.petsc import OptionsManager

class SNES_TimeDependentPDEVariationalProblem(TimeDependentPDEVariationalProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, u0, t_init, t_final, is_fwd_linear=False, solver_params=None):
        """
        varf_handler class
        conds = [u0, fwd_bc, adj_bc] : initial condition, (essential) fwd_bc, (essential) adj_bc
        When Vh[STATE] is MixedFunctionSpace, bc's are lists of DirichletBC classes 
        """
        super().__init__(Vh, varf_handler, bc, bc0, u0, t_init, t_final, is_fwd_linear)
        self.solver_params = solver_params
        self.comm = self.mesh.mpi_comm()

    def solveFwd(self, out, x, verbose=False):
        """ Solve the possibly nonlinear time dependent Fwd Problem:
        Given a, find u such that
        \delta_p F(u,m,p;\hat_p) = 0 \for all \hat_p"""
        out.zero()

        if self.solverA is None:
            self.solverA = self._createLUSolver()

        u_old = dl.Function(self.Vh[STATE])
        u_old.vector().zero()
        u_old.vector().axpy(1., self.init_cond.vector())
        out.store(u_old.vector(), self.t_init)
        
        A = None
        b = None

        if self.is_fwd_linear:
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            du = dl.TrialFunction(self.Vh[STATE])
            dp = dl.TestFunction(self.Vh[ADJOINT])
            u_vec = self.generate_static_state()
            
            for t in self.times[1:]:
                
                if self.comm.rank == 0:
                    print(f"solving at time:\t{t}", flush=True)
                start = time.perf_counter()
                
                A_form = ufl.lhs(self.varf(du, u_old, m, dp, t))
                b_form = ufl.rhs(self.varf(du, u_old, m, dp, t))
                self._set_time(self.fwd_bc, t)
                if A is None:
                    A, b = dl.assemble_system(A_form, b_form, self.fwd_bc)
                else:
                    A.zero()
                    b.zero()
                    dl.assemble_system(A_form, b_form, self.fwd_bc, A_tensor=A, b_tensor=b)
                    
                self.solverA.set_operator(A)
                
                self.solverA.solve(u_vec, b)

                out.store(u_vec, t)
                u_old.vector().zero()
                u_old.vector().axpy(1., u_vec)
                
                if self.comm.rank == 0:
                    print(f"Time step took:\t{time.perf_counter()-start}", flush=True)
        else:
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            u = dl.Function(self.Vh[STATE])
            dp = dl.TestFunction(self.Vh[ADJOINT])
            u_vec = self.generate_static_state()

            u.assign(u_old)
            
            u_old_old = dl.Function(self.Vh[STATE])  # for 3-point extrapolation in time
            u_old_old.vector().axpy(1, u_old.vector())
            
            for i, t in enumerate(self.times[1:]):
                if verbose:
                    start = time.perf_counter()
                    if self.comm.rank == 0:
                        print(f"solving at time:\t{t}", flush=True)
                
                # Richardson exptrapolation for initial guess, u = 2u_old - u_old_old
                u.vector().zero()
                u.vector().axpy(2., u_old.vector())
                u.vector().axpy(-1., u_old_old.vector())
                
                # set up nonlinear problem
                res_form = self.varf(u, u_old, m, dp, t)
                self._set_time(self.fwd_bc, t)
                
                optmgr = OptionsManager(self.solver_params, "fwd")
                if i != 0:
                    # Turn off SNES/KSP view for all but the first time step.
                    # Only works if the user set these options in the solver_params.
                    optmgr.parameters.pop("snes_view", None)
                    optmgr.parameters.pop("ksp_view", None)
                
                nl_problem = SNES_VariationalProblem(res_form, u, self.fwd_bc)
                solver = SNES_VariationalSolver(nl_problem, self.comm, optmgr)
                
                niters, converged = solver.solve()
                solver.cleanup()
                
                if verbose:
                    if self.comm.rank == 0:
                        print(f"Time Step took:\t{niters} iterations.", flush=True)
                        print(f"Time step took:\t{time.perf_counter()-start}", flush=True)
                
                out.store(u.vector(), t)
                u_old_old.assign(u_old)
                u_old.assign(u)
