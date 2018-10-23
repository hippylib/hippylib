# Copyright (c) 2016-2018, The University of Texas at Austin & University of
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
# Software Foundation) version 2.0 dated June 1991.

from __future__ import absolute_import, division, print_function

import dolfin as dl
import numpy as np
from .PDEProblem import PDEProblem
from .timeDependentVector import TimeDependentVector
from .timeDependentOperator import TimeDependentOperator
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linalg import Transpose 
from ..utils.vector2function import vector2Function
from ..utils.checkDolfinVersion import dlversion


class _NonlinearProblem(dl.NonlinearProblem):
    def __init__(self, L, a, bc):
        dl.NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bc = bc
        self.reset_sparsity = True

    def F(self, b, x):
        dl.assemble(self.L, tensor=b)
        if self.bc is not None:
            for cond in self.bc:
                cond.apply(b, x)

    def J(self, A, x):
        dl.assemble(self.a, tensor=A)
        if self.bc is not None:
            for cond in self.bc:
                cond.apply(A)


class TimeDependentPDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, conds, t_init, t_final, dt, is_fwd_linear = False):
        """
        varf_handler class
        conds = [u0, fwd_bc, adj_bc] : initial condition, (essential) fwd_bc, (essential) adj_bc
        When Vh[STATE] is MixedFunctionSpace, bc's are lists of DirichletBC classes 
        """
        
        self.Vh = Vh
        self.varf = varf_handler

        if isinstance(conds[1], dl.DirichletBC):
            self.fwd_bc = [conds[1]]
        else:
            self.fwd_bc = conds[1]

        if isinstance(conds[2], dl.DirichletBC):
            self.adj_bc = [conds[2]]
        else:
            self.adj_bc = conds[2]

        self.mesh = self.Vh[STATE].mesh()
        self.init_cond = dl.interpolate(conds[0], self.Vh[STATE])
        self.t_init = t_init    
        self.t_final = t_final
        self.dt = dt
        self.times = np.arange(self.t_init, self.t_final+.5*self.dt, self.dt)

        self.linearize_x = None
        self.solverA = None
        self.solverAadj = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        self.build_operator = False

        self._A_update = True
        self._Aadj_update = True
        self._need_Raa_update = True
        self._need_C_update = True
        self._need_Wua_update = True

        self.is_fwd_linear = is_fwd_linear      
        if not is_fwd_linear:
            self.NewtonSolver= dl.NewtonSolver()
            self.NewtonSolver.parameters["absolute_tolerance"] = 1e-10
            self.NewtonSolver.parameters["relative_tolerance"] = 1e-5
            self.NewtonSolver.parameters["maximum_iterations"] = 25

        #TODO: Modify the timeDependentVector init() to get rid of this (we don't always have mass matrix in mixed problems)
        self.M  = dl.assemble(dl.inner(dl.TrialFunction(self.Vh[STATE]), dl.TestFunction(self.Vh[ADJOINT]))*dl.dx)

        self.C = TimeDependentOperator()
        self.Wua = TimeDependentOperator()

    def mass(self, u, a, p, t):
        return self.varf.mass(u, a, p, t)

    def stiff(self, u, a, p, t):
        return self.varf.stiff(u, a, p, t)

    def rhs(self, u, a, p, t, is_adj = False):
        varf = dl.Constant(1./self.dt)*self.mass(u, a, p, t) \
            + self.varf.rhs_form(a, p, t, is_adj = is_adj)
        return varf

    def res_varf(self, u, a, p, t, u_old = None, is_adj = False):
        if u_old:
            return dl.Constant(1./self.dt)*(self.mass(u, a, p, t) - self.mass(u_old, a, p, t)) \
                + self.stiff(u, a, p, t) - self.varf.rhs_form(a, p, t, is_adj = is_adj)
        else:
            return dl.Constant(1./self.dt)*self.mass(u, a, p, t) + self.stiff(u, a, p, t) \
                - self.varf.rhs_form(a, p, t, is_adj = is_adj)

    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.times)
            u.initialize(self.M, 1)
            a = dl.Function(self.Vh[PARAMETER]).vector()
            p = TimeDependentVector(self.times)
            p.initialize(self.M, 0)
            return [u, a, p]
        elif component == STATE:
            u = TimeDependentVector(self.times)
            u.initialize(self.M, 0)
            return u
        elif component == PARAMETER:
            return dl.Function(self.Vh[PARAMETER]).vector()
        elif component == ADJOINT:
            p = TimeDependentVector(self.times)
            p.initialize(self.M, 0)
            return p
        else:
            raise Exception('Incorrect vector component')

    def generate_state(self):
        """ return a time dependent vector in the shape of the state """
        return self.generate_vector(component=STATE)

    def generate_parameter(self):
        """ return a time dependent vector in the shape of the adjoint """
        return self.generate_vector(component=PARAMETER)

    def generate_adjoint(self):
        """ return a time dependent vector in the shape of the adjoint """
        return self.generate_vector(component=ADJOINT)

    def generate_static_state(self):
        """ return a time dependent vector in the shape of the state """
        u = dl.Vector()
        self.M.init_vector(u, 1)
        return u 

    def generate_static_adjoint(self):
        """ return a static vector in the shape of the adjoint """
        p = dl.Vector()
        self.M.init_vector(p, 0)
        return p
   
    def init_parameter(self, a):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        a.init( dummy.mpi_comm(), dummy.local_range() )

    def solveFwd(self, out, x, tol):
        """ Solve the possibly nonlinear time dependent Fwd Problem:
        Given a, find u such that
        \delta_p F(u,a,p;\hat_p) = 0 \for all \hat_p"""
        out.zero()

        if self.solverA is None:
            self.solverA = self._createLUSolver()

        u_old = self.generate_static_state()
        u_old.zero()
        u_old.axpy(1., self.init_cond.vector())
        out.store(u_old, self.t_init)

        if self.is_fwd_linear:
            u = dl.Function(self.Vh[STATE])
            a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.Function(self.Vh[ADJOINT])
            du = dl.TrialFunction(self.Vh[STATE])
            dp = dl.TestFunction(self.Vh[ADJOINT])
            u_vec = self.generate_static_state()

            A_form = dl.derivative(dl.derivative(self.res_varf(u, a, p, self.t_init + self.dt), u, du), p, dp) 
            A = dl.assemble(dl.Constant(self.dt)*A_form )
            [bc.apply(A) for bc in self.fwd_bc]
            self.solverA.set_operator(A)
            
            for t in self.times[1:]:
                b_form = dl.derivative(self.rhs(vector2Function(u_old, self.Vh[STATE]), a, p, t), p, dp ) 
                b = dl.assemble(dl.Constant(self.dt)*b_form )
                if self.fwd_bc:
                    for bc in self.fwd_bc:
                        try:
                            bc.function_arg.t = t
                        except:
                            pass
                    [bc.apply(b) for bc in self.fwd_bc]

                self.solverA.solve(u_vec, b)

                out.store(u_vec, t)
                u_old = u_vec.copy()
        else:
            # The forward problem is nonlinear
            t = self.t_init
            u_vec = self.generate_static_state()

            u = dl.Function(self.Vh[STATE])
            a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])

            u0_func = dl.Function(self.Vh[STATE])
            u0_func.vector().zero()
            u0_func.vector().axpy(1.0,u_old)

            u0prev_func = dl.Function(self.Vh[STATE])
            u0prev_func.vector().zero()
            u0prev_func.vector().axpy(1.0,u_old)

            res_form = self.res_varf(u, a, p, t, u_old=u0_func)
            J = dl.derivative(res_form, u)

            problem = _NonlinearProblem(res_form, J, self.fwd_bc)

            for i, t in enumerate(self.times[1:]):
                self.NewtonSolver.solve(problem, u.vector())

                out.store(u.vector(), t)
                u0prev_func.assign(u0_func)
                u0_func.assign(u)

                if i > 0:
                    u.vector().zero()
                    u.vector().axpy(2.0, u0_func.vector())
                    u.vector().axpy(-1.0, u0prev_func.vector())


    def solveAdj(self, out, x, adj_rhs, tol):
        """ Solve the linear time dependent Adj Problem: 
            Given a, u; find p such that
            \delta_u F(u,a,p;\hat_u) = 0 \for all \hat_u
        """
        out.zero()

        u = dl.Function(self.Vh[STATE])
        a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])

        if self.solverAadj is None:
            self.solverAadj = self._createLUSolver() 

        p_vec = self.generate_static_adjoint()
        p_old = self.generate_static_adjoint()
        p_old.zero()
        rhs_t = self.generate_static_state()
        rhs_t.zero()

        for t in reversed(self.times[1:]):
            adj_rhs.retrieve(rhs_t, t)
            x[STATE].retrieve(u.vector(), t)
            b_form = dl.derivative(self.mass(u, a, vector2Function(p_old, self.Vh[ADJOINT]), t), u, du) 
            b = dl.assemble(b_form)
            rhs_t.axpy(1., b)

            adj_form = dl.Constant(self.dt)*dl.lhs(dl.derivative(dl.derivative(self.res_varf(u,a,p,0.), u, du), p, dp))
            Aadj = dl.assemble(adj_form)
            
            if self.adj_bc:
                [bc.apply(Aadj, rhs_t) for bc in self.adj_bc]

            self.solverAadj.set_operator(Aadj)
            self.solverAadj.solve(p_vec, rhs_t)

            out.store(p_vec, t)
            p_old = p_vec.copy()


    def evalGradientParameter(self, x, out):
        """Given u,a,p; eval \delta_a F(u,a,p; \hat_a) \for all \hat_a """
        out.zero()
        out_t = out.copy()
        da = dl.TestFunction(self.Vh[PARAMETER])
        u = dl.Function(self.Vh[STATE])
        p = dl.Function(self.Vh[ADJOINT])
        u_old = self.generate_static_state()
        a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])

        for t in self.times[1:]:
            x[STATE].retrieve(u_old, t-self.dt)
            x[STATE].retrieve(u.vector(), t)
            x[ADJOINT].retrieve(p.vector(), t)

            res_form = dl.Constant(self.dt)*(self.res_varf(u, a, p, t, vector2Function(u_old, self.Vh[STATE])) )
            dl.assemble( dl.derivative(res_form, a, da), tensor=out_t)
            out.axpy(1., out_t)


    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        self.linearize_x = x
        if self.solver_fwd_inc == None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
        

    def solveIncremental(self, out, rhs, is_adj, mytol):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, a, find \tilde_u s.t.:
            \delta_{pu} F(u,a,p; \hat_p, \tilde_u) = rhs for all \hat_p.
            
            If is_adj = True:
            Solve the adj incremental system:
            Given u, a, find \tilde_p s.t.:
            \delta_{up} F(u,a,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        out.zero()
        if self.solver_fwd_inc == None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()

        u = dl.Function(self.Vh[STATE])
        a = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER]) 
        p = dl.Function(self.Vh[ADJOINT])

        if is_adj:
            rhs_t = self.generate_static_state()
            phat = self.generate_static_adjoint() 
            phat.zero()
            phat_old = phat.copy()

            for t in reversed(self.times[1:]):
                self.linearize_x[STATE].retrieve(u.vector(), t)
                self.linearize_x[ADJOINT].retrieve(p.vector(), t)
                
                # Define adj inc solver
                varf = dl.Constant(self.dt)*self.res_varf(u, a, p, t)
                A = dl.assemble(dl.derivative(dl.derivative(varf, u), p))
                
                rhs.retrieve(rhs_t, t)
                rhs_mass = dl.assemble(dl.derivative(self.mass(u,a,vector2Function(phat_old, self.Vh[ADJOINT]),t), u))
                rhs_t.axpy(1., rhs_mass)
                
                if self.adj_bc is not None:
                    [bc.apply(A, rhs_t) for bc in self.adj_bc]
                
                self.solver_adj_inc.set_operator(A)
                self.solver_adj_inc.solve(phat, rhs_t)
                
                phat_old = phat.copy()
                out.store(phat, t)
        else:
            rhs_t = self.generate_static_adjoint() 
            uhat = self.generate_static_state() 
            uhat.zero()
            uhat_old = uhat.copy()

            for t in self.times[1:]:
                self.linearize_x[STATE].retrieve(u.vector(), t)
                self.linearize_x[ADJOINT].retrieve(p.vector(), t)
                
                # Fwd inc solver
                varf = dl.Constant(self.dt)*self.res_varf(u,a,p,t)
                A = dl.assemble(dl.derivative(dl.derivative(varf, p), u))
                
                rhs.retrieve(rhs_t, t) 
                rhs_mass = dl.assemble(dl.derivative(self.mass(vector2Function(uhat_old, self.Vh[STATE]),a,p,t), p))
                rhs_t.axpy(1., rhs_mass)

                if self.adj_bc is not None:
                    [bc.apply(A, rhs_t) for bc in self.adj_bc]

                self.solver_fwd_inc.set_operator(A)
                self.solver_fwd_inc.solve(uhat, rhs_t)
                
                uhat_old = uhat.copy()
                out.store(uhat, t)
            

    def applyC(self, da, out, x = None):
        out.zero()
        x = self.linearize_x

        product = self.generate_static_adjoint() 
        u = dl.Function(self.Vh[STATE])
        a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        for t in self.times[1:]:
            x[STATE].retrieve(u.vector(), t)
            x[ADJOINT].retrieve(p.vector(), t)
            varf = dl.Constant(self.dt)*self.res_varf(u, a, p, t)
            C = dl.assemble(dl.lhs(dl.derivative(dl.derivative(varf, p), a)) )
            
            if self.adj_bc:
                [bc.zero(C) for bc in self.adj_bc]
            
            C.mult(da, product)
            out.store(product, t)


    def applyCt(self, dp, out):
        out.zero()
        x = self.linearize_x

        product = self.generate_parameter() 
        u = dl.Function(self.Vh[STATE])
        a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        dp_t = self.generate_static_adjoint() 
        
        for t in self.times[1:]:
            x[STATE].retrieve(u.vector(), t)
            x[ADJOINT].retrieve(p.vector(), t)
            varf = dl.Constant(self.dt)*self.res_varf(u, a, p, t)
            C = dl.assemble(dl.lhs(dl.derivative(dl.derivative(varf, p), a)) )
            if self.adj_bc is not None:
                [bc.zero(C) for bc in self.adj_bc]
            dp.retrieve(dp_t, t)
            C.transpmult(dp_t, product)
            out.axpy(1., product)


    def applyWuu(self, du, out, gn_approx = False):
        out.zero()

        product = dl.Function(self.Vh[STATE]).vector()
        u = dl.Function(self.Vh[STATE])
        a = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        du_t = self.generate_static_state()

        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            varf = dl.Constant(self.dt)*self.res_varf(u, a, p, t)
            Wuu = dl.assemble(dl.derivative(dl.derivative(varf, u), u))
            
            if self.adj_bc is not None:
                [bc.zero(Wuu) for bc in self.adj_bc]
            
            Wuu_t = Transpose(Wuu)
            if self.adj_bc is not None:
                [bc.zero(Wuu_t) for bc in self.adj_bc]
            
            Wuu = Transpose(Wuu_t)
            du.retrieve(du_t, t)
            Wuu.mult(du_t, product)
            out.store(product, t)


    def applyWua(self, da, out):
        out.zero()

        product = self.generate_static_state()
        u = dl.Function(self.Vh[STATE])
        a = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])        
        
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            varf = dl.Constant(self.dt)*self.res_varf(u, a, p, t)
            Wau = dl.assemble(dl.lhs(dl.derivative(dl.derivative(varf, a), u)) )
            Wua = Transpose(Wau)
            if self.adj_bc is not None:
                [bc.zero(Wua) for bc in self.adj_bc]
            Wua.mult(da, product)
            out.store(product, t)


    def applyWau(self, du, out):
        out.zero()

        product = self.generate_parameter() 
        u = dl.Function(self.Vh[STATE])
        a = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        du_t = self.generate_static_state() 
        
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            varf = dl.Constant(self.dt)*self.res_varf(u, a, p, t)
            Wau = dl.assemble(dl.lhs(dl.derivative(dl.derivative(varf, a), u)) )
            Wua = Transpose(Wau)
            if self.adj_bc is not None:
                [bc.zero(Wua) for bc in self.adj_bc]
            Wau = Transpose(Wua)
    
            du.retrieve(du_t, t)
            Wau.mult(du_t, product)
            out.axpy(1., product)


    def applyWaa(self, da, out):
        out.zero()

        product = self.generate_parameter()
        u = dl.Function(self.Vh[STATE])
        a = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])        
        
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            varf = dl.Constant(self.dt)*self.res_varf(u, a, p, t)
            Waa = dl.assemble(dl.lhs(dl.derivative(dl.derivative(varf, a), a)) )
            Waa.mult(da, product)
            out.axpy(1., product)

 
    def apply_ij(self,i,j, dir, out):   
        """
            Given u, a, p; compute 
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE,STATE] = self.applyWuu
        KKT[PARAMETER, STATE] = self.applyWau
        KKT[STATE, PARAMETER] = self.applyWua
        KKT[PARAMETER, PARAMETER] = self.applyWaa
        KKT[ADJOINT, STATE] = None 
        KKT[STATE, ADJOINT] = None 
        
        KKT[ADJOINT, PARAMETER] = self.applyC
        KKT[PARAMETER, ADJOINT] = self.applyCt
        KKT[i,j](dir, out)
            

    def _createLUSolver(self):
        if dlversion() <= (1,6,0):
            return dl.PETScLUSolver()
        else:
            return dl.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )