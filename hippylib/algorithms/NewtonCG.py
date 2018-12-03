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

import math
from ..utils.parameterList import ParameterList
from ..modeling.reducedHessian import ReducedHessian
from ..modeling.variables import STATE, PARAMETER, ADJOINT
from .cgsolverSteihaug import CGSolverSteihaug


def LS_ParameterList():
    """
    Generate a ParameterList for line search globalization.
    type: :code:`LS_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [10, "Maximum number of backtracking iterations"]
    
    return ParameterList(parameters)

def TR_ParameterList():
    """
    Generate a ParameterList for Trust Region globalization.
    type: :code:`RT_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["eta"] = [0.05, "Reject step if (actual reduction)/(predicted reduction) < eta"]
    
    return ParameterList(parameters)

def ReducedSpaceNewtonCG_ParameterList():
    """
    Generate a ParameterList for ReducedSpaceNewtonCG.
    type: :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-18, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [20, "maximum number of iterations"]
    parameters["inner_rel_tolerance"]   = [1e-9, "relative tolerance used for the solution of the forward, adjoint, and incremental (fwd,adj) problems"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["GN_iter"]               = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    parameters["LS"]                    = [LS_ParameterList(), "Sublist containing LS globalization parameters"]
    parameters["TR"]                    = [TR_ParameterList(), "Sublist containing TR globalization parameters"]
    
    return ParameterList(parameters)
  
    

class ReducedSpaceNewtonCG:
    
    """
    Inexact Newton-CG method to solve constrained optimization problems in the reduced parameter space.
    The Newton system is solved inexactly by early termination of CG iterations via Eisenstat-Walker
    (to prevent oversolving) and Steihaug (to avoid negative curvature) criteria.
    Globalization is performed using one of the following methods:

    - line search (LS) based on the armijo sufficient reduction condition; or
    - trust region (TR) based on the prior preconditioned norm of the update direction.

    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.
       
    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient and the Hessian.
    
    More specifically the model object should implement following methods:
    
       - :code:`generate_vector()` -> generate the object containing state, parameter, adjoint
       - :code:`cost(x)` -> evaluate the cost functional, report regularization part and misfit separately
       - :code:`solveFwd(out, x,tol)` -> solve the possibly non linear forward problem up a tolerance :code:`tol`
       - :code:`solveAdj(out, x,tol)` -> solve the linear adjoint problem
       - :code:`evalGradientParameter(x, out)` -> evaluate the gradient of the parameter and compute its norm
       - :code:`setPointForHessianEvaluations(x)` -> set the state to perform hessian evaluations
       - :code:`solveFwdIncremental(out, rhs, tol)` -> solve the linearized forward problem for a given :code:`rhs`
       - :code:`solveAdjIncremental(out, rhs, tol)` -> solve the linear adjoint problem for a given :code:`rhs`
       - :code:`applyC(dm, out)`    --> Compute out :math:`= C_x dm`
       - :code:`applyCt(dp, out)`   --> Compute out = :math:`C_x  dp`
       - :code:`applyWuu(du,out)`   --> Compute out = :math:`(W_{uu})_x  du`
       - :code:`applyWmu(dm, out)`  --> Compute out = :math:`(W_{um})_x  dm`
       - :code:`applyWmu(du, out)`  --> Compute out = :math:`W_{mu}  du`
       - :code:`applyR(dm, out)`    --> Compute out = :math:`R  dm`
       - :code:`applyWmm(dm,out)`   --> Compute out = :math:`W_{mm} dm`
       - :code:`Rsolver()`          --> A solver for the regularization term
       
    Type :code:`help(Model)` for additional information
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, dm) less than tolerance"       #3
                           ]
    
    def __init__(self, model, parameters=ReducedSpaceNewtonCG_ParameterList(), callback = None):
        """
        Initialize the ReducedSpaceNewtonCG.
        Type :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for list of default parameters
        and their descriptions.
        
        Parameters:
        :code:`model` The model object that describes the inverse problem
        :code:`parameters`: (type :code:`ParameterList`, optional) set parameters for inexact Newton CG
        :code:`callback`: (type function handler with signature :code:`callback(it: int, x: list of dl.Vector): --> None`
               optional callback function to be called at the end of each iteration. Takes as input the iteration number, and
               the list of vectors for the state, parameter, adjoint.
        """
        self.model = model
        self.parameters = parameters
        
        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0
        
        self.callback = callback
        
    def solve(self, x):
        """

        Input: 
            :code:`x = [u, m, p]` represents the initial guess (u and p may be None). 
            :code:`x` will be overwritten on return.
        """
        if self.model is None:
            raise TypeError("model can not be of type None.")
        
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
            
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
            
        if self.parameters["globalization"] == "LS":
            return self._solve_ls(x)
        elif self.parameters["globalization"] == "TR":
            return self._solve_tr(x)
        else:
            raise ValueError(self.parameters["globalization"])
        
    def _solve_ls(self,x):
        """
        Solve the constrained optimization problem with initial guess :code:`x`.
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        
        c_armijo = self.parameters["LS"]["c_armijo"]
        max_backtracking_iter = self.parameters["LS"]["max_backtracking_iter"]
        
        self.model.solveFwd(x[STATE], x, innerTol)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        mhat = self.model.generate_vector(PARAMETER)    
        mg = self.model.generate_vector(PARAMETER)
                
        x_star = [None, None, None] + x[3::]
        x_star[STATE]     = self.model.generate_vector(STATE)
        x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
        
        cost_old, _, _ = self.model.cost(x)
        
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(x[ADJOINT], x, innerTol)
            
            self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            gradnorm = self.model.evalGradientParameter(x, mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            
            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            HessApply = ReducedHessian(self.model, innerTol)
            solver = CGSolverSteihaug(comm = self.model.prior.R.mpi_comm())
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Rsolver())
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(mhat, -mg)
            self.total_cg_iter += HessApply.ncalls
            
            alpha = 1.0
            descent = 0
            n_backtrack = 0
            
            mg_mhat = mg.inner(mhat)
            
            while descent == 0 and n_backtrack < max_backtracking_iter:
                # update m and u
                x_star[PARAMETER].zero()
                x_star[PARAMETER].axpy(1., x[PARAMETER])
                x_star[PARAMETER].axpy(alpha, mhat)
                x_star[STATE].zero()
                x_star[STATE].axpy(1., x[STATE])
                self.model.solveFwd(x_star[STATE], x_star, innerTol)
                
                cost_new, reg_new, misfit_new = self.model.cost(x_star)
                  
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + alpha * c_armijo * mg_mhat) or (-mg_mhat <= self.parameters["gdm_tolerance"]):
                    cost_old = cost_new
                    descent = 1
                    x[PARAMETER].zero()
                    x[PARAMETER].axpy(1., x_star[PARAMETER])
                    x[STATE].zero()
                    x[STATE].axpy(1., x_star[STATE])
                else:
                    n_backtrack += 1
                    alpha *= 0.5
                            
            if(print_level >= 0) and (self.it == 1):
                print( "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:14} {7:14} {8:14}".format(
                      "It", "cg_it", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "alpha", "tolcg") )
                
            if print_level >= 0:
                print( "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e}".format(
                        self.it, HessApply.ncalls, cost_new, misfit_new, reg_new, mg_mhat, gradnorm, alpha, tolcg) )
                
            if self.callback:
                self.callback(self.it, x)
                
                
            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
            
            if -mg_mhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new
        return x
    
    def _solve_tr(self,x):
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        
        eta_TR = self.parameters["TR"]["eta"]
        delta_TR = None
        
        
        self.model.solveFwd(x[STATE], x, innerTol)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        mhat = self.model.generate_vector(PARAMETER) 
        R_mhat = self.model.generate_vector(PARAMETER)   

        mg = self.model.generate_vector(PARAMETER)
        
        x_star = [None, None, None] + x[3::]
        x_star[STATE]     = self.model.generate_vector(STATE)
        x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
        
        cost_old, reg_old, misfit_old = self.model.cost(x)
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(x[ADJOINT], x, innerTol)
            
            self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            gradnorm = self.model.evalGradientParameter(x, mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            

            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            HessApply = ReducedHessian(self.model, innerTol)
            solver = CGSolverSteihaug(comm = self.model.prior.R.mpi_comm())
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Rsolver())
            if self.it > 1:
                solver.set_TR(delta_TR, self.model.prior.R)
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(mhat, -mg)
            self.total_cg_iter += HessApply.ncalls

            if self.it == 1:
                self.model.prior.R.mult(mhat,R_mhat)
                mhat_Rnorm = R_mhat.inner(mhat)
                delta_TR = max(math.sqrt(mhat_Rnorm),1)

            x_star[PARAMETER].zero()
            x_star[PARAMETER].axpy(1., x[PARAMETER])
            x_star[PARAMETER].axpy(1., mhat)   #m_star = m +mhat
            x_star[STATE].zero()
            x_star[STATE].axpy(1., x[STATE])      #u_star = u
            self.model.solveFwd(x_star[STATE], x_star, innerTol)
            cost_star, reg_star, misfit_star = self.model.cost(x_star)
            ACTUAL_RED = cost_old - cost_star
            #Calculate Predicted Reduction
            H_mhat = self.model.generate_vector(PARAMETER)
            H_mhat.zero()
            HessApply.mult(mhat,H_mhat)
            mg_mhat = mg.inner(mhat)
            PRED_RED = -0.5*mhat.inner(H_mhat) - mg_mhat
            # print( "PREDICTED REDUCTION", PRED_RED, "ACTUAL REDUCTION", ACTUAL_RED)
            rho_TR = ACTUAL_RED/PRED_RED


            # Nocedal and Wright Trust Region conditions (page 69)
            if rho_TR < 0.25:
                delta_TR *= 0.5
            elif rho_TR > 0.75 and solver.reasonid == 3:
                delta_TR *= 2.0
            

            # print( "rho_TR", rho_TR, "eta_TR", eta_TR, "rho_TR > eta_TR?", rho_TR > eta_TR , "\n")
            if rho_TR > eta_TR:
                x[PARAMETER].zero()
                x[PARAMETER].axpy(1.0,x_star[PARAMETER])
                x[STATE].zero()
                x[STATE].axpy(1.0,x_star[STATE])
                cost_old = cost_star
                reg_old = reg_star
                misfit_old = misfit_star
                accept_step = True
            else:
                accept_step = False
                
            if self.callback:
                self.callback(self.it, x)
                
                            
            if(print_level >= 0) and (self.it == 1):
                print( "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:14} {7:14} {8:14} {9:11} {10:14}".format(
                      "It", "cg_it", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "TR Radius", "rho_TR", "Accept Step","tolcg") )
                
            if print_level >= 0:
                print( "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e} {9:11} {10:14e}".format(
                        self.it, HessApply.ncalls, cost_old, misfit_old, reg_old, mg_mhat, gradnorm, delta_TR, rho_TR, accept_step,tolcg) )
                

            #TR radius can make this term arbitrarily small and prematurely exit.
            if -mg_mhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_old
        return x
