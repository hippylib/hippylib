# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
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

import dolfin as dl
import ufl
import numpy as np
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linalg import Transpose 
from ..algorithms.linSolvers import PETScLUSolver
from ..utils.vector2function import vector2Function
from .timeDependentVector import TimeDependentVector

class PDEProblem(object):
    """ Consider the PDE problem:
        Given :math:`m`, find :math:`u` such that 
        
            .. math:: F(u, m, p) = ( f(u, m), p) = 0, \\quad \\forall p.
        
        Here :math:`F` is linear in :math:`p`, but it may be non linear in :math:`u` and :math:`m`.
    """

    def generate_state(self):
        """ Return a vector in the shape of the state. """
        raise NotImplementedError("Child class should implement method generate_state")

    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        raise NotImplementedError("Child class should implement method generate_parameter")

    def init_parameter(self, m):
        """ Initialize the parameter. """
        raise NotImplementedError("Child class should implement method init_parameter")

    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that

            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0, \\quad \\forall \\hat{p}.
        """
        raise NotImplementedError("Child class should implement method solveFwd")

    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear adjoint problem: 
            Given :math:`m`, :math:`u`; find :math:`p` such that
            
                .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        raise NotImplementedError("Child class should implement method solveAdj")

    def evalGradientParameter(self, x, out):
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        raise NotImplementedError("Child class should implement method evalGradientParameter")
 
    def setLinearizationPoint(self,x, gauss_newton_approx):

        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. 
            Set whether Gauss Newton approximation of
            the Hessian should be used."""
        raise NotImplementedError("Child class should implement method setLinearizationPoint")
      
    def solveIncremental(self, out, rhs, is_adj):
        """ If :code:`is_adj = False`:

            Solve the forward incremental system:
            Given :math:`u, m`, find :math:`\\tilde{u}` such that

            .. math::
                \\delta_{pu} F(u, m, p; \\hat{p}, \\tilde{u}) = \\mbox{rhs}, \\quad \\forall \\hat{p}.
            
            If :code:`is_adj = True`:
            
            Solve the adjoint incremental system:
            Given :math:`u, m`, find :math:`\\tilde{p}` such that

            .. math::
                \\delta_{up} F(u, m, p; \\hat{u}, \\tilde{p}) = \\mbox{rhs}, \\quad \\forall \\hat{u}.
        """
        raise NotImplementedError("Child class should implement method solveIncremental")

    def apply_ij(self,i,j, dir, out):   
        """
            Given :math:`u, m, p`; compute 
            :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`, 
            :math:`\\forall \\hat{i}.`
        """
        raise NotImplementedError("Child class should implement method apply_ij")
        
    def apply_ijk(self,i,j,k, x, jdir, kdir, out):
        """
            Given :code:`x = [u,a,p]`; compute
            :math:`\\delta_{ijk} F(u,a,p; \\hat{i}, \\tilde{j}, \\tilde{k})`
            in the direction :math:`(\\tilde{j},\\tilde{k}) = (`:code:`jdir,kdir`), :math:`\\forall \\hat{i}.`
        """
        raise NotImplementedError("Child class should implement apply_ijk")

class PDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, is_fwd_linear = False):
        self.Vh = Vh
        self.varf_handler = varf_handler
        if type(bc) is dl.DirichletBC:
            self.bc = [bc]
        else:
            self.bc = bc
        if type(bc0) is dl.DirichletBC:
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0
        
        self.A  = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None
        
        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        
        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {"forward": 0,
                        "adjoint":0 ,
                        "incremental_forward":0,
                        "incremental_adjoint":0}
    def generate_state(self):
        """ Return a vector in the shape of the state. """
        return dl.Function(self.Vh[STATE]).vector()
    
    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return dl.Function(self.Vh[PARAMETER]).vector()
    
    def init_parameter(self, m):
        """ Initialize the parameter. """
        dummy = self.generate_parameter()
        m.init( dummy.mpi_comm(), dummy.local_range() )
    
    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
        
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""
        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
        if self.is_fwd_linear:
            u = dl.TrialFunction(self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u, m, p)
            A_form = ufl.lhs(res_form)
            b_form = ufl.rhs(res_form)
            A, b = dl.assemble_system(A_form, b_form, bcs=self.bc)
            self.solver.set_operator(A)
            self.solver.solve(state, b)
        else:
            u = vector2Function(x[STATE], self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u, m, p)
            dl.solve(res_form == 0, u, self.bc)
            state.zero()
            state.axpy(1., u.vector())
        
    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear adjoint problem: 
            Given :math:`m, u`; find :math:`p` such that
            
                .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
            
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p)
        adj_form = dl.derivative( dl.derivative(varf, u, du), p, dp )
        Aadj, dummy = dl.assemble_system(adj_form, ufl.inner(u,du)*ufl.dx, self.bc0)
        self.solver.set_operator(Aadj)
        self.solver.solve(adj, adj_rhs)
     
    def evalGradientParameter(self, x, out):
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = dl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)
        out.zero()
        dl.assemble( dl.derivative(res_form, m, dm), tensor=out)
         
    def setLinearizationPoint(self,x, gauss_newton_approx):
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """
            
        x_fun = [vector2Function(x[i], self.Vh[i]) for i in range(3)]
        
        f_form = self.varf_handler(*x_fun)
        
        g_form = [None,None,None]
        for i in range(3):
            g_form[i] = dl.derivative(f_form, x_fun[i])
            
        self.A, dummy = dl.assemble_system(dl.derivative(g_form[ADJOINT],x_fun[STATE]), g_form[ADJOINT], self.bc0)
        self.At, dummy = dl.assemble_system(dl.derivative(g_form[STATE],x_fun[ADJOINT]),  g_form[STATE], self.bc0)
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],x_fun[PARAMETER]))
        [bc.zero(self.C) for bc in self.bc0]
                
        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
        
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if gauss_newton_approx:
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
        else:
            self.Wuu = dl.assemble(dl.derivative(g_form[STATE],x_fun[STATE]))
            [bc.zero(self.Wuu) for bc in self.bc0]
            Wuu_t = Transpose(self.Wuu)
            [bc.zero(Wuu_t) for bc in self.bc0]
            self.Wuu = Transpose(Wuu_t)
            self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[STATE]))
            Wmu_t = Transpose(self.Wmu)
            [bc.zero(Wmu_t) for bc in self.bc0]
            self.Wmu = Transpose(Wmu_t)
            self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[PARAMETER]))
        
    def solveIncremental(self, out, rhs, is_adj):
        """ If :code:`is_adj == False`:

            Solve the forward incremental system:
            Given :math:`u, m`, find :math:`\\tilde{u}` such that
            
                .. math:: \\delta_{pu} F(u, m, p; \\hat{p}, \\tilde{u}) = \\mbox{rhs},\\quad \\forall \\hat{p}.
            
            If :code:`is_adj == True`:

            Solve the adjoint incremental system:
            Given :math:`u, m`, find :math:`\\tilde{p}` such that
            
                .. math:: \\delta_{up} F(u, m, p; \\hat{u}, \\tilde{p}) = \\mbox{rhs},\\quad \\forall \\hat{u}.
        """
        if is_adj:
            self.n_calls["incremental_adjoint"] += 1
            self.solver_adj_inc.solve(out, rhs)
        else:
            self.n_calls["incremental_forward"] += 1
            self.solver_fwd_inc.solve(out, rhs)
    
    def apply_ij(self,i,j, dir, out):   
        """
            Given :math:`u, m, p`; compute 
            :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`,
            :math:`\\forall \\hat{i}`.
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C
        
        if i >= j:
            if KKT[i,j] is None:
                out.zero()
            else:
                KKT[i,j].mult(dir, out)
        else:
            if KKT[j,i] is None:
                out.zero()
            else:
                KKT[j,i].transpmult(dir, out)
                
    def apply_ijk(self,i,j,k, x, jdir, kdir, out):
        x_fun = [vector2Function(x[ii], self.Vh[ii]) for ii in range(3)]
        idir_fun = dl.TestFunction(self.Vh[i])
        jdir_fun = vector2Function(jdir, self.Vh[j])
        kdir_fun = vector2Function(kdir, self.Vh[k])
        
        res_form = self.varf_handler(*x_fun)
        form = dl.derivative(
               dl.derivative(
               dl.derivative(res_form, x_fun[i], idir_fun),
               x_fun[j], jdir_fun),
               x_fun[k], kdir_fun)
        
        out.zero()
        dl.assemble(form, tensor=out)
        
        if i in [STATE,ADJOINT]:
            [bc.apply(out) for bc in self.bc0]
                   
    def _createLUSolver(self):   
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )

        
class TimeDependentPDEVariationalProblem(PDEProblem):
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

        #TODO: Modify the TimeDependentVector init() to get rid of this (we don't always have mass matrix in mixed problems)
        self.M  = dl.assemble(dl.inner(dl.TrialFunction(self.Vh[STATE]), dl.TestFunction(self.Vh[ADJOINT]))*dl.dx)


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
        with  dl.XDMFFile(fname) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False
            for t in self.times[1:]:
                u.retrieve(ufun.vector(), t)
                fid.write(ufun, t)


    def _createLUSolver(self):
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )
