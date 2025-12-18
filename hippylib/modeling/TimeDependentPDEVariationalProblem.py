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

import dolfin as dl

try:
    import ufl_legacy as ufl
except:
    import ufl
import numpy as np
from .PDEProblem import PDEProblem
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linalg import Transpose 
from ..algorithms.linSolvers import PETScLUSolver
from ..utils.vector2function import vector2Function
from ..utils.deprecate import deprecated
from .timeDependentVector import TimeDependentVector

@deprecated("ImplicitEulerTimeDependentPDEVariationalProblem", "3.3.0", "Please use TimeDependentPDEVariationalProblem instead unless you specifically need this class.")
class ImplicitEulerTimeDependentPDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, u0, t_init, t_final, is_fwd_linear = False):
        """
        varf_handler class
        conds = [u0, fwd_bc, adj_bc] : initial condition, (essential) fwd_bc, (essential) adj_bc
        When Vh[STATE] is MixedFunctionSpace, bc's are lists of DirichletBC classes 
        """

        self.Vh = Vh
        self.varf = varf_handler

        if isinstance(bc, dl.DirichletBC):
            self.fwd_bc = [bc]
        else:
            self.fwd_bc = bc

        if isinstance(bc0, dl.DirichletBC):
            self.adj_bc = [bc0]
        else:
            self.adj_bc = bc0

        self.mesh = self.Vh[STATE].mesh()
        self.init_cond = u0
        self.t_init = t_init    
        self.t_final = t_final
        self.dt = varf_handler.dt
        self.times = np.arange(self.t_init, self.t_final+.5*self.dt, self.dt)

        self.linearize_x = None
        self.solverA = None
        self.solverAadj = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear
        self.parameters = dl.NonlinearVariationalSolver.default_parameters()
        #self.parameters['nonlinear_solver'] = 'snes'
        #self.parameters['snes_solver']["absolute_tolerance"] = 1e-10
        #self.parameters['snes_solver']["relative_tolerance"] = 1e-5
        #self.parameters['snes_solver']["maximum_iterations"] = 100
        #self.parameters['snes_solver']["report"] = True


    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.times)
            u.initialize(self.Vh[STATE])
            a = dl.Function(self.Vh[PARAMETER]).vector()
            p = TimeDependentVector(self.times)
            p.initialize(self.Vh[ADJOINT])
            return [u, a, p]
        elif component == STATE:
            u = TimeDependentVector(self.times)
            u.initialize(self.Vh[STATE])
            return u
        elif component == PARAMETER:
            return dl.Function(self.Vh[PARAMETER]).vector()
        elif component == ADJOINT:
            p = TimeDependentVector(self.times)
            p.initialize(self.Vh[ADJOINT])
            return p
        else:
            raise Exception('Incorrect vector component')

    def generate_state(self):
        """ return a time dependent vector in the shape of the state """
        return self.generate_vector(component=STATE)

    def generate_parameter(self):
        """ return a vector in the shape of the parameter """
        return self.generate_vector(component=PARAMETER)

    def generate_adjoint(self):
        """ return a time dependent vector in the shape of the adjoint """
        return self.generate_vector(component=ADJOINT)

    def generate_static_state(self):
        """ Return a static vector in the shape of the parameter. """
        return dl.Function(self.Vh[STATE]).vector()

    def generate_static_adjoint(self):
        """ return a static vector in the shape of the adjoint """
        return dl.Function(self.Vh[ADJOINT]).vector()

    def init_parameter(self, a):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        a.init( dummy.mpi_comm(), dummy.local_range() )
        
    def _set_time(self, bcs, t):
        for bc in bcs:
            try:
                bc.function_arg.t = t
            except:
                pass
        

    def solveFwd(self, out, x):
        """ Solve the possibly nonlinear time dependent Fwd Problem:
        Given a, find u such that
        \delta_p F(u,m,p;\hat_p) = 0 \for all \hat_p"""
        out.zero()

        if self.solverA is None:
            self.solverA = self._createLUSolver()

        u_old = dl.Function(self.Vh[STATE])
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
        else:
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            u = dl.Function(self.Vh[STATE])
            dp = dl.TestFunction(self.Vh[ADJOINT])
            u_vec = self.generate_static_state()

            u.assign(u_old)
            for t in self.times[1:]:
                res_form = self.varf(u, u_old, m, dp, t)
                self._set_time(self.fwd_bc, t)
                dl.solve(res_form == 0, u, self.fwd_bc, solver_parameters=self.parameters)
                out.store(u.vector(), t)
                u_old.assign(u)
                

    def solveAdj(self, out, x, adj_rhs):
        """ Solve the linear time dependent Adj Problem: 
            Given a, u; find p such that
            \delta_u F(u,m,p;\hat_u) = 0 \for all \hat_u
        """
        out.zero()

        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])

        if self.solverAadj is None:
            self.solverAadj = self._createLUSolver() 

        p_vec = self.generate_static_adjoint()
        p_old = dl.Function(self.Vh[ADJOINT])
        rhs_t = self.generate_static_state()
        rhs_t.zero()
        
        Aadj = None
        b    = None

        for t in reversed(self.times[1:]):
            adj_rhs.retrieve(rhs_t, t)
            x[STATE].retrieve(u.vector(), t)
            
            form = self.varf(u, u_old, m, p, t)
            adj_form = dl.derivative(dl.derivative(form, u, du), p, dp)
            b_form = dl.Constant(-1.)*dl.derivative(dl.derivative(form, u_old, du), p, p_old)
            if Aadj is None:
                Aadj, b = dl.assemble_system(adj_form, b_form, self.adj_bc)
            else:
                Aadj.zero()
                b.zero()
                dl.assemble_system(adj_form, b_form, self.adj_bc, A_tensor=Aadj, b_tensor=b)
            b.axpy(1., rhs_t)

            self.solverAadj.set_operator(Aadj)
            self.solverAadj.solve(p_vec, b)

            out.store(p_vec, t)
            p_old.vector().zero()
            p_old.vector().axpy(1., p_vec)


    def evalGradientParameter(self, x, out):
        """Given u,m,p; eval \delta_m F(u,m,p; \hat_m) \for all \hat_m """
        out.zero()
        out_t = out.copy()
        dm = dl.TestFunction(self.Vh[PARAMETER])
        u = dl.Function(self.Vh[STATE])
        p = dl.Function(self.Vh[ADJOINT])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        
        x[STATE].retrieve(u_old.vector(), self.times[0])

        for t in self.times[1:]:
            x[STATE].retrieve(u.vector(), t)
            x[ADJOINT].retrieve(p.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            out_t.zero()
            dl.assemble( dl.derivative(form, m, dm), tensor=out_t)
            out.axpy(1., out_t)


    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        self.linearize_x = x
        self.gauss_newton_approx = gauss_newton_approx
        if self.solver_fwd_inc == None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
            

    def _solveIncrementalFwd(self, out, rhs):
        out.zero()
        u     = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER]) 
        
        dp = dl.TestFunction(self.Vh[ADJOINT])
        du = dl.TrialFunction(self.Vh[STATE])
        
        uhat = self.generate_static_state() 
        uhat.zero()
        
        uhat_old = dl.Function(self.Vh[STATE])   
        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            form = self.varf(u, u_old, m, dp, t)
            Ainc_form = dl.derivative(form, u, du)
            binc_form = dl.Constant(-1.)*dl.derivative(form, u_old, uhat_old)
            Ainc, binc = dl.assemble_system(Ainc_form, binc_form, self.adj_bc)
            binc.axpy(1., rhs.view(t))

            self.solver_fwd_inc.set_operator(Ainc)
            self.solver_fwd_inc.solve(uhat, binc)
            
            uhat_old.vector().zero()
            uhat_old.vector().axpy(1., uhat)
            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            
            out.store(uhat, t)

  
    def _solveIncrementalAdj(self, out, rhs):
        out.zero()
        u     = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p     = dl.Function(self.Vh[ADJOINT])
        p_old = dl.Function(self.Vh[ADJOINT])
        
        dp     = dl.TrialFunction(self.Vh[ADJOINT])
        
        du     = dl.TestFunction(self.Vh[STATE])
        du_old = dl.TestFunction(self.Vh[STATE])
        
        
        phat = self.generate_static_adjoint() 
        phat.zero()
        phat_old = dl.Function(self.Vh[ADJOINT])

        for it, t in enumerate( reversed(self.times[1:]) ):
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), self.times[it-1])
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            form = self.varf(u, u_old, m, p, t)
            A_adj_form = dl.derivative( dl.derivative(form, u, du), p, dp)
            b_adj_form  = dl.Constant(-1.)*dl.derivative( dl.derivative(form, u_old, du_old), p, phat_old)
            A_adj_inc, b_adj_inc = dl.assemble_system(A_adj_form, b_adj_form, self.adj_bc)
            b_adj_inc.axpy(1., rhs.view(t))
            self.solver_adj_inc.set_operator(A_adj_inc)
            self.solver_adj_inc.solve(phat, b_adj_inc)

            phat_old.vector().zero()
            phat_old.vector().axpy(1., phat)
            p_old.vector().zero()
            p_old.vector().axpy(1., p.vector() )
            
            out.store(phat, t)


    def solveIncremental(self, out, rhs, is_adj):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, a, find \tilde_u s.t.:
            \delta_{pu} F(u,a,p; \hat_p, \tilde_u) = rhs for all \hat_p.
            
            If is_adj = True:
            Solve the adj incremental system:
            Given u, a, find \tilde_p s.t.:
            \delta_{up} F(u,a,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            return self._solveIncrementalAdj(out, rhs)
        else:
            return self._solveIncrementalFwd(out, rhs)


    def applyC(self, dm, out):
        out.zero()

        out_t = self.generate_static_adjoint() 
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        dp = dl.TestFunction(self.Vh[ADJOINT])
        
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            cvarf = dl.derivative(dl.derivative(form, p, dp), m, dm_fun)
            out_t.zero()
            dl.assemble(cvarf, tensor=out_t)
            [bc.apply(out_t) for bc in self.adj_bc]

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            out.store(out_t, t)


    def applyCt(self, dp, out):
        out.zero()
        out_t = self.generate_parameter()
        
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        dm = dl.TestFunction(self.Vh[PARAMETER])
        
        dp_fun = dl.Function(self.Vh[ADJOINT])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            dp.retrieve(dp_fun.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            cvarf_adj = dl.derivative(dl.derivative(form, p, dp_fun), m, dm)
            out_t.zero()
            dl.assemble(cvarf_adj, tensor=out_t)

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            
            out.axpy(1., out_t)


    def applyWuu(self, du, out):

        out.zero()
        
        if self.gauss_newton_approx == True:
            return
        
        u     = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])

        du_fun  = dl.Function(self.Vh[STATE])
        du_old  = dl.Function(self.Vh[STATE])
        
        du_test  = dl.TestFunction(self.Vh[STATE])
        du_old_test  = dl.TestFunction(self.Vh[STATE])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        du.retrieve(du_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            du.retrieve(du_fun.vector(), t)
            
            form  = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, u, du_fun), u, du_test) + \
                   dl.derivative(dl.derivative(form, u_old, du_old), u_old, du_old_test)
                   
            out_t = dl.assemble(varf)
            [bc.apply(out_t) for bc in self.adj_bc]
            
            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            du.retrieve(du_old.vector(), t)

            out.store(out_t, t)


    def applyWum(self, dm, out):
        out.zero()
        
        if self.gauss_newton_approx == True:
            return
        
        u = dl.Function(self.Vh[STATE])
        u_old  = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        dm_fun = vector2Function(dm, self.Vh[PARAMETER]) 
        
        du_test  = dl.TestFunction(self.Vh[STATE])
        du_old_test  = dl.TestFunction(self.Vh[STATE])
        
        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])    

        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            form  = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, m, dm_fun), u, du_test) + \
                   dl.derivative(dl.derivative(form, m, dm_fun), u_old, du_old_test)
                   
            out_t = dl.assemble(varf)
            [bc.apply(out_t) for bc in self.adj_bc]

            self.linearize_x[STATE].retrieve(u_old.vector(), t)   
            out.store(out_t, t)


    def applyWmu(self, du, out):
        out.zero()
        
        if self.gauss_newton_approx == True:
            return
        
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        du_fun  = dl.Function(self.Vh[STATE])
        du_old  = dl.Function(self.Vh[STATE])
        
        dm_test = dl.TestFunction(self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        du.retrieve(du_old.vector(), self.times[0])
        
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            du.retrieve(du_fun.vector(), t)
            
            form  = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, u, du_fun), m, dm_test) + \
                   dl.derivative(dl.derivative(form, u_old, du_old), m, dm_test)
                   
            out_t = dl.assemble(varf)
            
            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            du.retrieve(du_old.vector(), t)

            out.axpy(1., out_t)


    def applyWmm(self, dm, out):
        out.zero()
        
        if self.gauss_newton_approx == True:
            return

        out_t = self.generate_parameter()
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])   
        
        dm_fun = vector2Function( dm, self.Vh[PARAMETER])
        
        dm_test = dl.TestFunction(self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            form = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, m, dm_fun), m, dm_test)
            out_t.zero()
            dl.assemble(varf, tensor=out_t)

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            out.axpy(1., out_t)


    def apply_ij(self,i,j, dir, out): 
        """
            Given u, a, p; compute 
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE,STATE] = self.applyWuu
        KKT[PARAMETER, STATE] = self.applyWmu
        KKT[STATE, PARAMETER] = self.applyWum
        KKT[PARAMETER, PARAMETER] = self.applyWmm
        KKT[ADJOINT, STATE] = None 
        KKT[STATE, ADJOINT] = None 

        KKT[ADJOINT, PARAMETER] = self.applyC
        KKT[PARAMETER, ADJOINT] = self.applyCt
        KKT[i,j](dir, out)
        
    def exportState(self, u, fname):
        ufun = dl.Function(self.Vh[STATE], name="state")
        with dl.XDMFFile(self.mesh.mpi_comm(), fname) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False
            
            # write the initial condition to file first.
            ufun.vector().zero()
            ufun.vector().axpy(1., self.init_cond.vector())
            fid.write(ufun, self.times[0])
            
            # retrieve the snapshots and write those to file.
            for t in self.times[1:]:
                u.retrieve(ufun.vector(), t)
                fid.write(ufun, t)


    def _createLUSolver(self):
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )


class TimeDependentPDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, u0, t_init, t_final, is_fwd_linear = False):
        """
        Time-dependent PDE problem class for generic one-step methods. The time-stepping scheme is implemented but the :code:varf_handler`, which defines the variation form relating the state at the current and previous time step,

        ..math::`r(u_n, u_{n-1}, m, v) = 0 \qquad \forall v.`

        conds = [u0, fwd_bc, adj_bc] : initial condition, (essential) fwd_bc, (essential) adj_bc
        When Vh[STATE] is MixedFunctionSpace, bc's are lists of DirichletBC classes 
        """

        self.Vh = Vh
        self.varf = varf_handler

        if isinstance(bc, dl.DirichletBC):
            self.fwd_bc = [bc]
        else:
            self.fwd_bc = bc

        if isinstance(bc0, dl.DirichletBC):
            self.adj_bc = [bc0]
        else:
            self.adj_bc = bc0

        self.mesh = self.Vh[STATE].mesh()
        self.init_cond = u0
        self.t_init = t_init    
        self.t_final = t_final
        self.dt = varf_handler.dt
        self.times = np.arange(self.t_init, self.t_final+.5*self.dt, self.dt)

        self.linearize_x = None
        self.solverA = None
        self.solverAadj = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear
        self.parameters = dl.NonlinearVariationalSolver.default_parameters()
        #self.parameters['nonlinear_solver'] = 'snes'
        #self.parameters['snes_solver']["absolute_tolerance"] = 1e-10
        #self.parameters['snes_solver']["relative_tolerance"] = 1e-5
        #self.parameters['snes_solver']["maximum_iterations"] = 100
        #self.parameters['snes_solver']["report"] = True


    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.times)
            u.initialize(self.Vh[STATE])
            a = dl.Function(self.Vh[PARAMETER]).vector()
            p = TimeDependentVector(self.times)
            p.initialize(self.Vh[ADJOINT])
            return [u, a, p]
        elif component == STATE:
            u = TimeDependentVector(self.times)
            u.initialize(self.Vh[STATE])
            return u
        elif component == PARAMETER:
            return dl.Function(self.Vh[PARAMETER]).vector()
        elif component == ADJOINT:
            p = TimeDependentVector(self.times)
            p.initialize(self.Vh[ADJOINT])
            return p
        else:
            raise Exception('Incorrect vector component')

    def generate_state(self):
        """ return a time dependent vector in the shape of the state """
        return self.generate_vector(component=STATE)

    def generate_parameter(self):
        """ return a vector in the shape of the parameter """
        return self.generate_vector(component=PARAMETER)

    def generate_adjoint(self):
        """ return a time dependent vector in the shape of the adjoint """
        return self.generate_vector(component=ADJOINT)

    def generate_static_state(self):
        """ Return a static vector in the shape of the parameter. """
        return dl.Function(self.Vh[STATE]).vector()

    def generate_static_adjoint(self):
        """ return a static vector in the shape of the adjoint """
        return dl.Function(self.Vh[ADJOINT]).vector()

    def init_parameter(self, a):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        a.init( dummy.mpi_comm(), dummy.local_range() )
        
    def _set_time(self, bcs, t):
        for bc in bcs:
            try:
                bc.function_arg.t = t
            except:
                pass
        

    def solveFwd(self, out, x):
        """ Solve the possibly nonlinear time dependent Fwd Problem:
        Given a, find u such that
        \delta_p F(u,m,p;\hat_p) = 0 \for all \hat_p"""
        out.zero()

        if self.solverA is None:
            self.solverA = self._createLUSolver()

        u_old = dl.Function(self.Vh[STATE])
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
        else:
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            u = dl.Function(self.Vh[STATE])
            dp = dl.TestFunction(self.Vh[ADJOINT])
            u_vec = self.generate_static_state()

            u.assign(u_old)
            for t in self.times[1:]:
                res_form = self.varf(u, u_old, m, dp, t)
                self._set_time(self.fwd_bc, t)
                dl.solve(res_form == 0, u, self.fwd_bc, solver_parameters=self.parameters)
                out.store(u.vector(), t)
                u_old.assign(u)
                

    def solveAdj(self, out, x, adj_rhs):
        """ Solve the linear time dependent Adj Problem: 
            Given a, u; find p such that
            \delta_u F(u,m,p;\hat_u) = 0 \for all \hat_u
        """
        out.zero()
        u_next = dl.Function(self.Vh[STATE]) # u_{n+1}
        u = dl.Function(self.Vh[STATE]) # u_n
        u_old = dl.Function(self.Vh[STATE]) # u_{n-1}
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])

        if self.solverAadj is None:
            self.solverAadj = self._createLUSolver() 

        p_vec = self.generate_static_adjoint()
        p_old = dl.Function(self.Vh[ADJOINT]) # p_{n+1}
        rhs_t = self.generate_static_state()
        rhs_t.zero()
        
        Aadj = None
        b    = None

        is_initial_step = True
        t_next = self.times[-1]

        for t, t_old in zip(reversed(self.times[1:]), reversed(self.times[:-1])):
            adj_rhs.retrieve(rhs_t, t)
            x[STATE].retrieve(u.vector(), t)
            x[STATE].retrieve(u_old.vector(), t_old)

            form = self.varf(u, u_old, m, p, t)
            adj_form = dl.derivative(dl.derivative(form, u, du), p, dp)

            if is_initial_step:
                # No contribution from u_{n+1} at the final time step
                b_form = dl.Constant(0.0) * du * ufl.dx 
                is_initial_step = False
            else:
                # Contribution from u_{n+1} at all other time steps
                next_form = self.varf(u_next, u, m, p_old, t_next)
                b_form = dl.Constant(-1.)*dl.derivative(next_form, u, du)

            if Aadj is None:
                Aadj, b = dl.assemble_system(adj_form, b_form, self.adj_bc)
            else:
                Aadj.zero()
                b.zero()
                dl.assemble_system(adj_form, b_form, self.adj_bc, A_tensor=Aadj, b_tensor=b)
            b.axpy(1., rhs_t)

            self.solverAadj.set_operator(Aadj)
            self.solverAadj.solve(p_vec, b)

            out.store(p_vec, t)

            # Update vectors
            p_old.vector().zero()
            p_old.vector().axpy(1., p_vec)

            u_next.vector().zero()
            u_next.vector().axpy(1., u.vector())
            t_next = t 


    def evalGradientParameter(self, x, out):
        """Given u,m,p; eval \delta_m F(u,m,p; \hat_m) \for all \hat_m """
        out.zero()
        out_t = out.copy()
        dm = dl.TestFunction(self.Vh[PARAMETER])
        u = dl.Function(self.Vh[STATE])
        p = dl.Function(self.Vh[ADJOINT])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        
        x[STATE].retrieve(u_old.vector(), self.times[0])

        for t in self.times[1:]: 
            x[STATE].retrieve(u.vector(), t)
            x[ADJOINT].retrieve(p.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            out_t.zero()
            dl.assemble( dl.derivative(form, m, dm), tensor=out_t)
            out.axpy(1., out_t)
            x[STATE].retrieve(u_old.vector(), t)


    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        self.linearize_x = x
        self.gauss_newton_approx = gauss_newton_approx
        if self.solver_fwd_inc == None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
            

    def _solveIncrementalFwd(self, out, rhs):
        out.zero()
        u     = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER]) 
        
        dp = dl.TestFunction(self.Vh[ADJOINT])
        du = dl.TrialFunction(self.Vh[STATE])
        
        uhat = self.generate_static_state() 
        uhat.zero()
        
        uhat_old = dl.Function(self.Vh[STATE])   
        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            form = self.varf(u, u_old, m, dp, t)
            Ainc_form = dl.derivative(form, u, du)
            binc_form = dl.Constant(-1.)*dl.derivative(form, u_old, uhat_old)
            Ainc, binc = dl.assemble_system(Ainc_form, binc_form, self.adj_bc)
            binc.axpy(1., rhs.view(t))

            self.solver_fwd_inc.set_operator(Ainc)
            self.solver_fwd_inc.solve(uhat, binc)
            
            uhat_old.vector().zero()
            uhat_old.vector().axpy(1., uhat)
            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            
            out.store(uhat, t)

  
    def _solveIncrementalAdj(self, out, rhs):
        out.zero()
        u     = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        u_next = dl.Function(self.Vh[STATE])

        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p     = dl.Function(self.Vh[ADJOINT])
        p_old = dl.Function(self.Vh[ADJOINT])
        
        dp     = dl.TrialFunction(self.Vh[ADJOINT])
        
        du     = dl.TestFunction(self.Vh[STATE])
        du_old = dl.TestFunction(self.Vh[STATE])
        
        
        phat = self.generate_static_adjoint() 
        phat.zero()
        phat_old = dl.Function(self.Vh[ADJOINT])

        is_initial_step = True
        t_next = self.times[-1]

        for t, t_old in zip( reversed(self.times[1:]), reversed(self.times[:-1]) ):
            # print(t, t_old)
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_old)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            form = self.varf(u, u_old, m, p, t)
            A_adj_form = dl.derivative( dl.derivative(form, u, du), p, dp)

            if is_initial_step:
                # No contribution from u_{n+1} at the final time step
                b_adj_form = dl.Constant(0.0) * du * ufl.dx
                is_initial_step = False
            else:   
                # Contribution from u_{n+1} at all other time steps
                next_form = self.varf(u_next, u, m, phat_old, t_next)
                b_adj_form = dl.Constant(-1.)*dl.derivative(next_form, u, du)

            # b_adj_form  = dl.Constant(-1.)*dl.derivative( dl.derivative(form, u_old, du_old), p, phat_old)
            A_adj_inc, b_adj_inc = dl.assemble_system(A_adj_form, b_adj_form, self.adj_bc)
            b_adj_inc.axpy(1., rhs.view(t))
            self.solver_adj_inc.set_operator(A_adj_inc)
            self.solver_adj_inc.solve(phat, b_adj_inc)

            out.store(phat, t)

            # Update vectors 
            phat_old.vector().zero()
            phat_old.vector().axpy(1., phat)
            p_old.vector().zero()
            p_old.vector().axpy(1., p.vector() )
            u_next.vector().zero()
            u_next.vector().axpy(1., u.vector())
            t_next = t 
            

    def solveIncremental(self, out, rhs, is_adj):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, a, find \tilde_u s.t.:
            \delta_{pu} F(u,a,p; \hat_p, \tilde_u) = rhs for all \hat_p.
            
            If is_adj = True:
            Solve the adj incremental system:
            Given u, a, find \tilde_p s.t.:
            \delta_{up} F(u,a,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            return self._solveIncrementalAdj(out, rhs)
        else:
            return self._solveIncrementalFwd(out, rhs)


    def applyC(self, dm, out):
        out.zero()

        out_t = self.generate_static_adjoint() 
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        dp = dl.TestFunction(self.Vh[ADJOINT])
        
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            cvarf = dl.derivative(dl.derivative(form, p, dp), m, dm_fun)
            out_t.zero()
            dl.assemble(cvarf, tensor=out_t)
            [bc.apply(out_t) for bc in self.adj_bc]

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            out.store(out_t, t)


    def applyCt(self, dp, out):
        out.zero()
        out_t = self.generate_parameter()
        
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        dm = dl.TestFunction(self.Vh[PARAMETER])
        
        dp_fun = dl.Function(self.Vh[ADJOINT])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            dp.retrieve(dp_fun.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            cvarf_adj = dl.derivative(dl.derivative(form, p, dp_fun), m, dm)
            out_t.zero()
            dl.assemble(cvarf_adj, tensor=out_t)

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            
            out.axpy(1., out_t)


    def applyWuu(self, du, out):

        out.zero()
        
        if self.gauss_newton_approx == True:
            return
        
        u     = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])

        du_fun  = dl.Function(self.Vh[STATE])
        du_old  = dl.Function(self.Vh[STATE])
        
        du_test  = dl.TestFunction(self.Vh[STATE])
        du_old_test  = dl.TestFunction(self.Vh[STATE])



        for it, t in enumerate(self.times[1:]):
            t_index = it + 1 # because we start from times[1:]

            # past-to-current step 
            t_old = self.times[t_index - 1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_old)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            du.retrieve(du_fun.vector(), t)

            form  = self.varf(u, u_old, m, p, t)

            # current input dir to current output dir 
            varf_t = dl.derivative(dl.derivative(form, u, du_fun), u, du_test)
            out_t = dl.assemble(varf_t)
            out.view(t).axpy(1.0, out_t)

            # current input dir to previous output dir 
            varf_t_old = dl.derivative(dl.derivative(form, u, du_fun), u_old, du_test)
            out_t_old = dl.assemble(varf_t_old)
            out.view(t_old).axpy(1.0, out_t_old)


            if t_index < len(self.times) - 1:
                # current-to-next step 
                t_next = self.times[t_index + 1]
                self.linearize_x[STATE].retrieve(u.vector(), t_next)
                self.linearize_x[STATE].retrieve(u_old.vector(), t)
                self.linearize_x[ADJOINT].retrieve(p.vector(), t_next)

                # current input dir to current output dir 
                varf_t = dl.derivative(dl.derivative(form, u_old, du_fun), u_old, du_test)
                out_t = dl.assemble(varf_t)
                out.view(t).axpy(1.0, out_t)

                # current input dir to next output dir 
                varf_t_next = dl.derivative(dl.derivative(form, u_old, du_fun), u, du_test)
                out_t_next = dl.assemble(varf_t_next)
                out.view(t_next).axpy(1.0, out_t_next)

        for t in self.times:
            [bc.apply(out.view(t)) for bc in self.adj_bc]


    def applyWum(self, dm, out):
        out.zero()
        
        if self.gauss_newton_approx == True:
            return
        
        u = dl.Function(self.Vh[STATE])
        u_old  = dl.Function(self.Vh[STATE])

        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])

        p = dl.Function(self.Vh[ADJOINT])
        p_old = dl.Function(self.Vh[ADJOINT])
        
        dm_fun = vector2Function(dm, self.Vh[PARAMETER]) 
        
        du_test  = dl.TestFunction(self.Vh[STATE])
        du_old_test  = dl.TestFunction(self.Vh[STATE])
        
        # self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])    

        # print(len(self.times[1:] - 1))

        for it, t in enumerate(self.times[1:]):
            t_index = it + 1 # because we start from times[1:]

            # past-to-current step 
            t_old = self.times[t_index - 1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_old)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            form  = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, m, dm_fun), u, du_test)

            out_t = dl.assemble(varf)
            if t_index < len(self.times) - 1:
                # current-to-next step 
                t_next = self.times[t_index + 1]
                self.linearize_x[STATE].retrieve(u.vector(), t_next)
                self.linearize_x[STATE].retrieve(u_old.vector(), t)
                self.linearize_x[ADJOINT].retrieve(p.vector(), t_next)

                form  = self.varf(u, u_old, m, p, t)
                varf = dl.derivative(dl.derivative(form, m, dm_fun), u_old, du_test)
                out_t_next = dl.assemble(varf)
                out_t.axpy(1.0, out_t_next)

            [bc.apply(out_t) for bc in self.adj_bc]

            self.linearize_x[STATE].retrieve(u_old.vector(), t)   
            out.store(out_t, t)


    def applyWmu(self, du, out):
        out.zero()
        
        if self.gauss_newton_approx == True:
            return
        
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        du_fun  = dl.Function(self.Vh[STATE])
        du_old  = dl.Function(self.Vh[STATE])
        
        dm_test = dl.TestFunction(self.Vh[PARAMETER])

        # self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        # du.retrieve(du_old.vector(), self.times[0])
        # print(len(self.times))


        for it, t in enumerate(self.times[1:]):
            t_index = it + 1 # because we start from times[1:]

            # past-to-current step 
            t_old = self.times[t_index - 1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_old)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            du.retrieve(du_fun.vector(), t)

            form  = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, u, du_fun), m, dm_test)
            out_t = dl.assemble(varf)

            if t_index < len(self.times) - 1:
                # current-to-next step
                t_next = self.times[t_index + 1]
                self.linearize_x[STATE].retrieve(u.vector(), t_next)
                self.linearize_x[STATE].retrieve(u_old.vector(), t)
                self.linearize_x[ADJOINT].retrieve(p.vector(), t_next)

                form  = self.varf(u, u_old, m, p, t_next)
                varf = dl.derivative(dl.derivative(form, u_old, du_fun), m, dm_test)
                out_t_next = dl.assemble(varf)
                out_t.axpy(1.0, out_t_next)

            out.axpy(1., out_t)


    def applyWmm(self, dm, out):
        out.zero()
        
        if self.gauss_newton_approx == True:
            return

        out_t = self.generate_parameter()
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])   
        
        dm_fun = vector2Function( dm, self.Vh[PARAMETER])
        
        dm_test = dl.TestFunction(self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            form = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, m, dm_fun), m, dm_test)
            out_t.zero()
            dl.assemble(varf, tensor=out_t)

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            out.axpy(1., out_t)


    def apply_ij(self,i,j, dir, out): 
        """
            Given u, a, p; compute 
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE,STATE] = self.applyWuu
        KKT[PARAMETER, STATE] = self.applyWmu
        KKT[STATE, PARAMETER] = self.applyWum
        KKT[PARAMETER, PARAMETER] = self.applyWmm
        KKT[ADJOINT, STATE] = None 
        KKT[STATE, ADJOINT] = None 

        KKT[ADJOINT, PARAMETER] = self.applyC
        KKT[PARAMETER, ADJOINT] = self.applyCt
        KKT[i,j](dir, out)
        
    def exportState(self, u, fname):
        ufun = dl.Function(self.Vh[STATE], name="state")
        with dl.XDMFFile(self.mesh.mpi_comm(), fname) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False
            
            # write the initial condition to file first.
            ufun.vector().zero()
            ufun.vector().axpy(1., self.init_cond.vector())
            fid.write(ufun, self.times[0])
            
            # retrieve the snapshots and write those to file.
            for t in self.times[1:]:
                u.retrieve(ufun.vector(), t)
                fid.write(ufun, t)


    def _createLUSolver(self):
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )



