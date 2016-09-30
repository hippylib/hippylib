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

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append( "../../" )
from hippylib import *


def u_boundary(x, on_boundary):
    return on_boundary

class Poisson:
    def __init__(self, mesh, Vh, Prior):
        """
        Construct a model by proving
        - the mesh
        - the finite element spaces for the STATE/ADJOINT variable and the PARAMETER variable
        - the Prior information
        """
        self.mesh = mesh
        self.Vh = Vh
        
        # Initialize Expressions
        self.atrue = Expression('log(2 + 7*(pow(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2),0.5) > 0.2))')
        self.f = Expression("1.0")
        self.u_o = Vector()
        
        self.u_bdr = Expression("0.0")
        self.u_bdr0 = Expression("0.0")
        self.bc = DirichletBC(self.Vh[STATE], self.u_bdr, u_boundary)
        self.bc0 = DirichletBC(self.Vh[STATE], self.u_bdr0, u_boundary)
        
        # Assemble constant matrices      
        self.Prior = Prior
        self.Wuu = self.assembleWuu()
        

        self.computeObservation(self.u_o)
                
        self.A = []
        self.At = []
        self.C = []
        self.Raa = []
        self.Wau = []
        
    def generate_vector(self, component="ALL"):
        """
        Return the list x=[u,a,p] where:
        - u is any object that describes the state variable
        - a is a Vector object that describes the parameter variable.
          (Need to support linear algebra operations)
        - p is any object that describes the adjoint variable
        
        If component is STATE, PARAMETER, or ADJOINT return x[component]
        """
        if component == "ALL":
            x = [Vector(), Vector(), Vector()]
            self.Wuu.init_vector(x[STATE],0)
            self.Prior.init_vector(x[PARAMETER],0)
            self.Wuu.init_vector(x[ADJOINT], 0)
        elif component == STATE:
            x = Vector()
            self.Wuu.init_vector(x,0)
        elif component == PARAMETER:
            x = Vector()
            self.Prior.init_vector(x,0)
        elif component == ADJOINT:
            x = Vector()
            self.Wuu.init_vector(x,0)
            
        return x
    
    def init_parameter(self, a):
        """
        Reshape a so that it is compatible with the parameter variable
        """
        self.Prior.init_vector(a,0)
        
    def assembleA(self,x, assemble_adjoint = False, assemble_rhs = False):
        """
        Assemble the matrices and rhs for the forward/adjoint problems
        """
        trial = TrialFunction(self.Vh[STATE])
        test = TestFunction(self.Vh[STATE])
        c = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        Avarf = inner(exp(c)*nabla_grad(trial), nabla_grad(test))*dx
        if not assemble_adjoint:
            bform = inner(self.f, test)*dx
            Matrix, rhs = assemble_system(Avarf, bform, self.bc)
        else:
            # Assemble the adjoint of A (i.e. the transpose of A)
            s = vector2Function(x[STATE], self.Vh[STATE])
            obs = vector2Function(self.u_o, self.Vh[STATE])
            bform = inner(obs - s, test)*dx
            Matrix, rhs = assemble_system(adjoint(Avarf), bform, self.bc0)
            
        if assemble_rhs:
            return Matrix, rhs
        else:
            return Matrix
    
    def assembleC(self, x):
        """
        Assemble the derivative of the forward problem with respect to the parameter
        """
        trial = TrialFunction(self.Vh[PARAMETER])
        test = TestFunction(self.Vh[STATE])
        s = vector2Function(x[STATE], Vh[STATE])
        c = vector2Function(x[PARAMETER], Vh[PARAMETER])
        Cvarf = inner(exp(c) * trial * nabla_grad(s), nabla_grad(test)) * dx
        C = assemble(Cvarf)
#        print "||c||", x[PARAMETER].norm("l2"), "||s||", x[STATE].norm("l2"), "||C||", C.norm("linf")
        self.bc0.zero(C)
        return C
       
    def assembleWuu(self):
        """
        Assemble the misfit operator
        """
        trial = TrialFunction(self.Vh[STATE])
        test = TestFunction(self.Vh[STATE])
        varf = inner(trial, test)*dx
        Wuu = assemble(varf)
        Wuu_t = Transpose(Wuu)
        self.bc0.zero(Wuu_t)
        Wuu = Transpose(Wuu_t)
        self.bc0.zero(Wuu)
        return Wuu
    
    def assembleWau(self, x):
        """
        Assemble the derivative of the parameter equation with respect to the state
        """
        trial = TrialFunction(self.Vh[STATE])
        test  = TestFunction(self.Vh[PARAMETER])
        a = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        c = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        varf = inner(exp(c)*nabla_grad(trial),nabla_grad(a))*test*dx
        Wau = assemble(varf)
        Wau_t = Transpose(Wau)
        self.bc0.zero(Wau_t)
        Wau = Transpose(Wau_t)
        return Wau
    
    def assembleRaa(self, x):
        """
        Assemble the derivative of the parameter equation with respect to the parameter (Newton method)
        """
        trial = TrialFunction(self.Vh[PARAMETER])
        test  = TestFunction(self.Vh[PARAMETER])
        s = vector2Function(x[STATE], self.Vh[STATE])
        c = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        a = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        varf = inner(nabla_grad(a),exp(c)*nabla_grad(s))*trial*test*dx
        return assemble(varf)

        
    def computeObservation(self, u_o):
        """
        Compute the syntetic observation
        """
        at = interpolate(self.atrue, Vh[PARAMETER])
        x = [self.generate_vector(STATE), at.vector(), None]
        A, b = self.assembleA(x, assemble_rhs = True)
        
        A.init_vector(u_o, 1)
        solve(A, u_o, b)
        
        # Create noisy data, ud
        MAX = u_o.norm("linf")
        noise = .01 * MAX * np.random.normal(0, 1, len(u_o.array()))
        u_o.set_local(u_o.array() + noise)
        plot(vector2Function(u_o, Vh[STATE]), title = "Observation")
    
    def cost(self, x):
        """
        Given the list x = [u,a,p] which describes the state, parameter, and
        adjoint variable compute the cost functional as the sum of 
        the misfit functional and the regularization functional.
        
        Return the list [cost functional, regularization functional, misfit functional]
        
        Note: p is not needed to compute the cost functional
        """        
        assert x[STATE] != None
                
        diff = x[STATE] - self.u_o
        Wuudiff = self.Wuu*diff
        misfit = .5 * diff.inner(Wuudiff)
        
        Rx = Vector()
        self.Prior.init_vector(Rx,0)
        self.Prior.R.mult(x[PARAMETER], Rx)
        reg = .5 * x[PARAMETER].inner(Rx)
        
        c = misfit + reg
        
        return c, reg, misfit
    
    def solveFwd(self, out, x, tol=1e-9):
        """
        Solve the forward problem.
        """
        A, b = self.assembleA(x, assemble_rhs = True)
        A.init_vector(out, 1)
        solver = PETScKrylovSolver("cg", amg_method())
        solver.parameters["relative_tolerance"] = tol
        solver.set_operator(A)
        nit = solver.solve(out,b)
        
#        print "FWD", (self.A*out - b).norm("l2")/b.norm("l2"), nit

    
    def solveAdj(self, out, x, tol=1e-9):
        """
        Solve the adjoint problem.
        """
        At, badj = self.assembleA(x, assemble_adjoint = True,assemble_rhs = True)
        At.init_vector(out, 1)
        
        solver = PETScKrylovSolver("cg", amg_method())
        solver.parameters["relative_tolerance"] = tol
        solver.set_operator(At)
        nit = solver.solve(out,badj)
        
#        print "ADJ", (self.At*out - badj).norm("l2")/badj.norm("l2"), nit
    
    def evalGradientParameter(self,x, mg):
        """
        Evaluate the gradient for the variation parameter equation at the point x=[u,a,p].
        Parameters:
        - x = [u,a,p] the point at which to evaluate the gradient.
        - mg the variational gradient (g, atest) being atest a test function in the parameter space
          (Output parameter)
        
        Returns the norm of the gradient in the correct inner product g_norm = sqrt(g,g)
        """ 
        C = self.assembleC(x)

        self.Prior.init_vector(mg,0)
        C.transpmult(x[ADJOINT], mg)
        Rx = Vector()
        self.Prior.init_vector(Rx,0)
        self.Prior.R.mult(x[PARAMETER], Rx)   
        mg.axpy(1., Rx)
        
        g = Vector()
        self.Prior.init_vector(g,1)
        
        self.Prior.Msolver.solve(g, mg)
        g_norm = sqrt( g.inner(mg) )
        
        return g_norm
        
    
    def setPointForHessianEvaluations(self, x):  
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        """      
        self.A  = self.assembleA(x)
        self.At = self.assembleA(x, assemble_adjoint=True )
        self.C  = self.assembleC(x)
        self.Wau = self.assembleWau(x)
        self.Raa = self.assembleRaa(x)

        
    def solveFwdIncremental(self, sol, rhs, tol):
        """
        Solve the incremental forward problem for a given rhs
        """
        solver = PETScKrylovSolver("cg", amg_method())
        solver.set_operator(self.A)
        solver.parameters["relative_tolerance"] = tol
        self.A.init_vector(sol,1)
        nit = solver.solve(sol,rhs)
#        print "FwdInc", (self.A*sol-rhs).norm("l2")/rhs.norm("l2"), nit
        
    def solveAdjIncremental(self, sol, rhs, tol):
        """
        Solve the incremental adjoint problem for a given rhs
        """
        solver = PETScKrylovSolver("cg", amg_method())
        solver.set_operator(self.At)
        solver.parameters["relative_tolerance"] = tol
        self.At.init_vector(sol,1)
        nit = solver.solve(sol, rhs)
#        print "AdjInc", (self.At*sol-rhs).norm("l2")/rhs.norm("l2"), nit
    
    def applyC(self, da, out):
        self.C.mult(da,out)
    
    def applyCt(self, dp, out):
        self.C.transpmult(dp,out)
    
    def applyWuu(self, du, out, gn_approx=False):
        self.Wuu.mult(du, out)
    
    def applyWua(self, da, out):
        self.Wau.transpmult(da,out)

    
    def applyWau(self, du, out):
        self.Wau.mult(du, out)
    
    def applyR(self, da, out):
        self.Prior.R.mult(da, out)
        
    def Rsolver(self):        
        return self.Prior.Rsolver
    
    def applyRaa(self, da, out):
        self.Raa.mult(da, out)
            
if __name__ == "__main__":
    set_log_active(False)
    nx = 64
    ny = 64
    mesh = UnitSquareMesh(nx, ny)
    Vh2 = FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    
    Prior = LaplacianPrior(Vh[PARAMETER], gamma=1e-8, delta=1e-9)
    model = Poisson(mesh, Vh, Prior)
        
    a0 = interpolate(Expression("sin(x[0])"), Vh[PARAMETER])
    modelVerify(model, a0.vector(), 1e-12)

    a0 = interpolate(Expression("0.0"),Vh[PARAMETER])
    solver = ReducedSpaceNewtonCG(model)
    solver.parameters["abs_tolerance"] = 1e-9
    solver.parameters["inner_rel_tolerance"] = 1e-15
    solver.parameters["c_armijo"] = 1e-4
    solver.parameters["GN_iter"] = 6
    
    x = solver.solve(a0.vector())
    
    if solver.converged:
        print "\nConverged in ", solver.it, " iterations."
    else:
        print "\nNot Converged"

    print "Termination reason: ", solver.termination_reasons[solver.reason]
    print "Final gradient norm: ", solver.final_grad_norm
    print "Final cost: ", solver.final_cost
    
    xx = [vector2Function(x[i], Vh[i]) for i in range(len(Vh))]
    plot(xx[STATE], title = "State")
    plot(exp(xx[PARAMETER]), title = "exp(Parameter)")
    plot(xx[ADJOINT], title = "Adjoint")
    #interactive()
    
    model.setPointForHessianEvaluations(x)
    Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], gauss_newton_approx=False, misfit_only=True)
    p = 50
    k = min( 250, Vh[PARAMETER].dim()-p)
    Omega = np.random.randn(x[PARAMETER].array().shape[0], k+p)
    d, U = singlePassG(Hmisfit, Prior.R, Prior.Rsolver, Omega, k)
    plt.figure()
    plt.plot(range(0,k), d, 'b*')
    plt.yscale('log')
    
    interactive()
    plt.show()
    
