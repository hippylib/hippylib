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

# todo: add preconditioner for jacobian

from typing import List
import dolfin as dl

try:
    import ufl_legacy as ufl
except:
    import ufl
from petsc4py import PETSc

from ..utils.petsc import getPETScReasons, SNESConvergenceError

KSPReasons = getPETScReasons(PETSc.KSP.ConvergedReason())
SNESReasons = getPETScReasons(PETSc.SNES.ConvergedReason())

def check_snes_convergence(snes):
    """Check the convergence reason(s) for a PETSc.SNES object.
    Modified from the firedrake project. https://github.com/firedrakeproject/firedrake
    """
    r = snes.getConvergedReason()
    try:
        reason = SNESReasons[r]
        inner = False
    except KeyError:
        r = snes.getKSP().getConvergedReason()
        try:
            inner = True
            reason = KSPReasons[r]
        except KeyError:
            reason = "unknown reason (petsc4py enum incomplete?), try with -snes_converged_reason and -ksp_converged_reason"
    if r < 0:
        if inner:
            msg = f"Inner linear solve failed to converge after {snes.getKSP().getIterationNumber()} iteration(s) with reason: {reason}"
        else:
            msg = reason
        raise SNESConvergenceError(f"Nonlinear solve failed to converge after {snes.getIterationNumber()} nonlinear iteration(s). Reason:\n{msg}")
    
    return reason


class SNES_VariationalProblem():
    """Direct use of PETSc SNES interface to solve
    Nonlinear Variation Problem F(u; v) = 0.
    ref: https://fenicsproject.discourse.group/t/dusing-petsc4py-petsc-snes-directly/2368/18
    ref: https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/4
    """
    def __init__(self, F:ufl.Form, u:dl.Function, bcs:List[dl.DirichletBC]):
        """Constructor.

        Args:
            F (ufl.Form): The nonlinear form (typically a PDE in residual form)
            u (dl.Function): The function to solve for.
            bcs (list): List of boundary conditions.
        """
        self.u = u                                  # initial iterate
        self.V = self.u.function_space()
        du = dl.TrialFunction(self.V)
        self.u_test = dl.TestFunction(self.V)
        self.L = F                                  # residual form of nonlinear problem
        self.J_form  = dl.derivative(F, u, du)      # Jacobian form
        self.bcs = bcs

    def evalFunction(self, snes, x, F):
        """Form the residual for this problem.

        Args:
            snes (PETSc.SNES): PETSc SNES object.
            x (PETSc.Vec): Current iterate.
            F (PETSc.Vec): Residual at current iterate.
        """
        x = dl.PETScVector(x)
        Fvec  = dl.PETScVector(F)
        
        x.vec().copy(self.u.vector().vec())     # copy PETSc iterate to dolfin
        self.u.vector().apply("")               # update ghost values
        dl.assemble(self.L, tensor=Fvec)        # assemble residual
        
        # apply boundary conditions
        for bc in self.bcs:
            bc.apply(Fvec, x)
            bc.apply(Fvec, self.u.vector())

    def evalJacobian(self, snes, x, J, P):
        """Form the residual for this problem.

        Args:
            snes (PETSc.SNES): PETSc SNES object.
            x (PETSc.Vec): Current iterate.
            F (PETSc.Vec): Residual at current iterate.
            P (PETSc.Vec): Preconditioner.
        """
        Jvec = dl.PETScMatrix(J)
        x.copy(self.u.vector().vec())       # copy PETSc iterate to dolfin
        self.u.vector().apply("")           # update ghost values
        dl.assemble(self.J_form, tensor=Jvec)  # assemble Jacobian

        # apply boundary conditions.
        for bc in self.bcs:
            bc.apply(Jvec)
            #  bc.apply(P)


class SNES_VariationalSolver():
    """Direct use of PETSc SNES interface to solve
    Nonlinear Variation Problem F(u; v) = 0.
    
    See the following forum posts for more information.
    ref: https://fenicsproject.discourse.group/t/dusing-petsc4py-petsc-snes-directly/2368/18
    ref: https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/4
    
    :code:`WARNING` There is a potential segfault issue with the :code:`b` vector in the :code:`SNES` object.
    ref: https://fenicsproject.discourse.group/t/setting-snes-solver-with-petsc4py-segfaults/5149/8
    """
    def __init__(self, problem, comm, optmgr):
        """Constructor.

        Args:
            problem (SNES_VariationalProblem): The nonlinear variational problem.
            comm: An MPI communicator.
            optmgr (hippylib.OptionsManager): Options manager for the SNES solver.
        """
        self.problem = problem
        self.comm = comm
        self.optmgr = optmgr

        # create SNES solver
        self.snes = PETSc.SNES().create(self.comm)
        optmgr.set_from_options(self.snes)

        # set function, jacobian
        b = dl.PETScVector()
        J_mat = dl.PETScMatrix()
        # P_mat = dl.PETScMatrix()
        self.snes.setFunction(self.problem.evalFunction, b.vec())
        self.snes.setJacobian(self.problem.evalJacobian, J_mat.mat())


    def solve(self):
        with self.optmgr.inserted_options():
            self.snes.solve(None, self.problem.u.vector().vec())
        return self.getIterationNumber(), self.getConvergedReason()


    def getConvergedReason(self):
        return check_snes_convergence(self.snes)


    def getIterationNumber(self):
        return self.snes.getIterationNumber()
    
    
    def cleanup(self):
        self.snes.destroy()  # destroy the SNES object to avoid memory leak
        