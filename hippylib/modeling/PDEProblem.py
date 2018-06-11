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
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linalg import Transpose 
from ..utils.vector2function import vector2Function
from ..utils.checkDolfinVersion import dlversion

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

    def solveFwd(self, state, x, tol):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that

            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0, \\quad \\forall \\hat{p}.
        """
        raise NotImplementedError("Child class should implement method solveFwd")

    def solveAdj(self, state, x, adj_rhs, tol):
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
      
    def solveIncremental(self, out, rhs, is_adj, mytol):
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
    
    def solveFwd(self, state, x, tol):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
        
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""
        if self.solver is None:
            self.solver = self._createLUSolver()
        if self.is_fwd_linear:
            u = dl.TrialFunction(self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u, m, p)
            A_form = dl.lhs(res_form)
            b_form = dl.rhs(res_form)
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
        
    def solveAdj(self, adj, x, adj_rhs, tol):
        """ Solve the linear adjoint problem: 
            Given :math:`m, u`; find :math:`p` such that
            
                .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        if self.solver is None:
            self.solver = self._createLUSolver()
            
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p)
        adj_form = dl.derivative( dl.derivative(varf, u, du), p, dp )
        Aadj, dummy = dl.assemble_system(adj_form, dl.inner(u,du)*dl.dx, self.bc0)
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
        
    def solveIncremental(self, out, rhs, is_adj, mytol):
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
            self.solver_adj_inc.solve(out, rhs)
        else:
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
        if dlversion() <= (1,6,0):
            return dl.PETScLUSolver()
        else:
            return dl.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )
