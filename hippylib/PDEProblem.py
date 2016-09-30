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
from variables import STATE, PARAMETER, ADJOINT
from linalg import Transpose, vector2Function

class PDEProblem:
    """ Consider the PDE Problem:
        Given a, find u s.t. 
        F(u,a,p) = ( f(u,a), p) = 0 for all p.
        Here F is linear in p, but it may be non-linear in u and a.
    """
        
    def generate_state(self):
        """ return a vector in the shape of the state """
        
    def generate_parameter(self):
        """ return a vector in the shape of the parameter """
        
    def init_parameter(self, a):
        """ initialize the parameter """
    
    def solveFwd(self, state, x, tol):
        """ Solve the possibly nonlinear Fwd Problem:
        Given a, find u such that
        \delta_p F(u,a,p;\hat_p) = 0 \for all \hat_p"""
        
    def solveAdj(self, state, x, adj_rhs, tol):
        """ Solve the linear Adj Problem: 
            Given a, u; find p such that
            \delta_u F(u,a,p;\hat_u) = 0 \for all \hat_u
        """
     
    def eval_da(self, x, out):
        """Given u,a,p; eval \delta_a F(u,a,p; \hat_a) \for all \hat_a """
         
    def setLinearizationPoint(self,x):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        
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
    
    def apply_ij(self,i,j, dir, out):   
        """
            Given u, a, p; compute 
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """

class PDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, is_fwd_linear = False):
        self.Vh = Vh
        self.varf_handler = varf_handler
        self.bc = bc
        self.bc0 = bc0
        
        self.A  = []
        self.At = []
        self.C = []
        self.Wau = []
        self.Waa = []
        self.Wuu = []
        
        self.solver_fwd_inc = dl.PETScLUSolver()
        self.solver_adj_inc = dl.PETScLUSolver()
        
        self.is_fwd_linear = is_fwd_linear
        
    def generate_state(self):
        """ return a vector in the shape of the state """
        return dl.Function(self.Vh[STATE]).vector()
    
    def generate_parameter(self):
        """ return a vector in the shape of the parameter """
        return dl.Function(self.Vh[PARAMETER]).vector()
    
    def init_parameter(self, a):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        a.init( dummy.mpi_comm(), dummy.local_range() )
    
    def solveFwd(self, state, x, tol):
        """ Solve the possibly nonlinear Fwd Problem:
        Given a, find u such that
        \delta_p F(u,a,p;\hat_p) = 0 \for all \hat_p"""
        if self.is_fwd_linear:
            u = dl.TrialFunction(self.Vh[STATE])
            a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u,a,p)
            A_form = dl.lhs(res_form)
            b_form = dl.rhs(res_form)
            A, b = dl.assemble_system(A_form, b_form, bcs=self.bc)
            solver = dl.PETScLUSolver()
            solver.set_operator(A)
            solver.solve(state, b)
        else:
            u = vector2Function(x[STATE], self.Vh[STATE])
            a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u,a,p)
            dl.solve(res_form == 0, u, self.bc)
            state.zero()
            state.axpy(1., u.vector())
        
    def solveAdj(self, adj, x, adj_rhs, tol):
        """ Solve the linear Adj Problem: 
            Given a, u; find p such that
            \delta_u F(u,a,p;\hat_u) = 0 \for all \hat_u
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u,a,p)
        adj_form = dl.derivative( dl.derivative(varf, u, du), p, dp )
        Aadj, dummy = dl.assemble_system(adj_form, dl.Constant(0.)*du*dl.dx, self.bc0)
        solver = dl.PETScLUSolver()
        solver.set_operator(Aadj)
        solver.solve(adj, adj_rhs)
     
    def eval_da(self, x, out):
        """Given u,a,p; eval \delta_a F(u,a,p; \hat_a) \for all \hat_a """
        u = vector2Function(x[STATE], self.Vh[STATE])
        a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        da = dl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u,a,p)
        out.zero()
        dl.assemble( dl.derivative(res_form, a, da), tensor=out)
         
    def setLinearizationPoint(self,x):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        u = vector2Function(x[STATE], self.Vh[STATE])
        a = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        x_fun = [u,a,p]
        
        f_form = self.varf_handler(u,a,p)
        
        g_form = [None,None,None]
        for i in range(3):
            g_form[i] = dl.derivative(f_form, x_fun[i])
            
        self.A, dummy = dl.assemble_system(dl.derivative(g_form[ADJOINT],u), g_form[ADJOINT], self.bc0)
        self.At, dummy = dl.assemble_system(dl.derivative(g_form[STATE],p),  g_form[STATE], self.bc0)
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],a))
        self.bc0.zero(self.C)
        self.Wau = dl.assemble(dl.derivative(g_form[PARAMETER],u))
        Wau_t = Transpose(self.Wau)
        self.bc0.zero(Wau_t)
        self.Wau = Transpose(Wau_t)
        
        self.Wuu = dl.assemble(dl.derivative(g_form[STATE],u))
        self.bc0.zero(self.Wuu)
        Wuu_t = Transpose(self.Wuu)
        self.bc0.zero(Wuu_t)
        self.Wuu = Transpose(Wuu_t)
        
        self.Waa = dl.assemble(dl.derivative(g_form[PARAMETER],a))
        
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)
        
                
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
        if is_adj:
            self.solver_fwd_inc.solve(out, rhs)
        else:
            self.solver_adj_inc.solve(out, rhs)
            
        
    
    def apply_ij(self,i,j, dir, out):   
        """
            Given u, a, p; compute 
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wau
        KKT[PARAMETER, PARAMETER] = self.Waa
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C
        
        if i >= j:
            KKT[i,j].mult(dir, out)
        else:
            KKT[j,i].transpmult(dir, out)

    