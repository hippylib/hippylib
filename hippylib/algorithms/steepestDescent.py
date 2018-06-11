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

from ..modeling.variables import STATE, PARAMETER, ADJOINT
from ..utils.parameterList import ParameterList

def SteepestDescent_ParameterList():
        parameters = {}
        parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
        parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
        parameters["max_iter"]              = [500, "maximum number of iterations"]
        parameters["inner_rel_tolerance"]   = [1e-9, "relative tolerance used for the solution of the forward, adjoint, and incremental (fwd,adj) problems"]
        parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
        parameters["max_backtracking_iter"] = [10, "Maximum number of backtracking iterations"]
        parameters["print_level"]           = [0, "Print info on screen"]
        parameters["alpha"]                 = [1., "Initial scaling alpha"]
        return ParameterList(parameters)

class SteepestDescent:
    
    """
    Prior-preconditioned Steepest Descent to solve constrained optimization problems in the reduced parameter space.
    Globalization is performed using the Armijo sufficient reduction condition (backtracking).
    The stopping criterion is based on a control on the norm of the gradient.
       
    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient.
    
    More specifically the model object should implement following methods:

       - :code:`generate_vector()` -> generate the object containing state, parameter, adjoint.
       - :code:`cost(x)` -> evaluate the cost functional, report regularization part and misfit separately.
       - :code:`solveFwd(out, x,tol)` -> solve the possibly non linear forward problem up to tolerance :code:`tol`.
       - :code:`solveAdj(out, x,tol)` -> solve the linear adjoint problem.
       - :code:`evalGradientParameter(x, out)` -> evaluate the gradient of the parameter and compute its norm.
       - :code:`Rsolver()`          --> A solver for the regularization term.
       
    Type :code:`help(Model)` for additional information.
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           ]
    
    def __init__(self, model, parameters = SteepestDescent_ParameterList()):
        """
        Initialize the Steepest Descent solver. Type :code:`SteepestDescent_ParameterList().showMe()` for list of default parameters
        and their descriptions.
        """
        self.model = model
        self.parameters = parameters
        
        self.it = 0
        self.converged = False
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0
        
    def solve(self,x):
        """
        Solve the constrained optimization problem with initial guess :code:`x = [u,a,p]`. Return the solution :code:`[u,a,p]`.

        .. note:: :code:`x` will be overwritten.
        
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        c_armijo = self.parameters["c_armijo"]
        max_backtracking_iter = self.parameters["max_backtracking_iter"]
        alpha = self.parameters["alpha"]
        print_level = self.parameters["print_level"]
        
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
            
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
                        
            self.model.Rsolver().solve(mhat, -mg)
            
            alpha *= 2.
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
                if (cost_new < cost_old + alpha * c_armijo * mg_mhat):
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
                print( "\n{0:3} {1:15} {2:15} {3:15} {4:15} {5:15}".format(
                      "It", "cost", "misfit", "reg", "||g||L2", "alpha") )
                
            if print_level >= 0:
                print( "{0:3d} {1:15e} {2:15e} {3:15e} {4:15e} {5:15e}".format(
                        self.it, cost_new, misfit_new, reg_new, gradnorm, alpha) )
                
            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
                                        
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new
        return x