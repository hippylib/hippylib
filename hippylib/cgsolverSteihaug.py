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

from dolfin import Vector
import math

class CGSolverSteihaug:
    """
    Solve the linear system A x = b using preconditioned conjugate gradient ( B preconditioner)
    and the Steihaug stopping criterion:
    - reason of termination 0: we reached the maximum number of iterations (no convergence)
    - reason of termination 1: we reduced the residual up to the given tolerance (convergence)
    - reason of termination 2: we reached a negative direction (premature termination due to not spd matrix)
    
    The stopping criterion is based on either
    - the absolute preconditioned residual norm check: || r^* ||_{B^{-1}} < atol
    - the relative preconditioned residual norm check: || r^* ||_{B^{-1}}/|| r^0 ||_{B^{-1}} < rtol
    where r^* = b - Ax^* is the residual at convergence and r^0 = b - Ax^0 is the initial residual.
    
    The operator A is set using the method set_operator(A).
    A must provide the following two methods:
    - A.mult(x,y): y = A*x
    - A.init_vector(x, dim): initialize the vector x so that it is compatible with the range (dim = 0) or
      the domain (dim = 1) of A.
      
    The preconditioner B is set using the method set_preconditioner(B).
    B must provide the following method:
    - B.solve(z,r): z is the action of the preconditioner B on the vector r
    
    To solve the linear system A*x = b call self.solve(x,b).
    Here x and b are assumed to be FEniCS::Vector objects.
    
    The parameter attributes allows to set:
    - rel_tolerance     : the relative tolerance for the stopping criterion
    - abs_tolerance     : the absolute tolerance for the stopping criterion
    - max_iter          : the maximum number of iterations
    - zero_initial_guess: if True we start with a 0 initial guess
                          if False we use the x as initial guess.
    - print_level       : verbosity level:
                          -1 --> no output on screen
                           0 --> only final residual at convergence
                                 or reason for not not convergence
    """

    reason = ["Maximum Number of Iterations Reached",
              "Relative/Absolute residual less than tol",
              "Reached a negative direction"
              ]
    def __init__(self):
        self.parameters = {}
        self.parameters["rel_tolerance"] = 1e-9
        self.parameters["abs_tolerance"] = 1e-12
        self.parameters["max_iter"]      = 1000
        self.parameters["zero_initial_guess"] = True
        self.parameters["print_level"] = 0
        
        self.A = None
        self.B = None
        self.converged = False
        self.iter = 0
        self.reasonid = 0
        self.final_norm = 0
        
        self.r = Vector()
        self.z = Vector()
        self.d = Vector()
                
    def set_operator(self, A):
        """
        Set the operator A.
        """
        self.A = A
        self.A.init_vector(self.r,0)
        self.A.init_vector(self.z,0)
        self.A.init_vector(self.d,0)
        
    def set_preconditioner(self, B):
        """
        Set the preconditioner B.
        """
        self.B = B
        
    def solve(self,x,b):
        """
        Solve the linear system Ax = b
        """
        self.iter = 0
        self.converged = False
        self.reasonid  = 0
        
        betanom = 0.0
        alpha = 0.0 
        beta = 0.0
                
        if self.parameters["zero_initial_guess"]:
            self.r.zero()
            self.r.axpy(1.0, b)
            x.zero()
        else:
            self.A.mult(x,self.r)
            self.r *= -1.0
            self.r.axpy(1.0, b)
        
        self.z.zero()
        self.B.solve(self.z,self.r) #z = B^-1 r  
              
        self.d.zero()
        self.d.axpy(1.,self.z); #d = z
        
        nom0 = self.d.inner(self.r)
        nom = nom0
        
        if self.parameters["print_level"] == 1:
            print " Iterartion : ", 0, " (B r, r) = ", nom
            
        rtol2 = nom * self.parameters["rel_tolerance"] * self.parameters["rel_tolerance"]
        atol2 = self.parameters["abs_tolerance"] * self.parameters["abs_tolerance"]
        r0 = max(rtol2, atol2)
        
        if nom <= r0:
            self.converged  = True
            self.reasonid   = 1
            self.final_norm = math.sqrt(nom)
            if(self.parameters["print_level"] >= 0):
                print self.reason[self.reasonid]
                print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
            return
        
        self.A.mult(self.d, self.z)  #z = A d
        den = self.z.inner(self.d)
        
        if den <= 0.0:
            self.converged = True
            self.reasonid = 2
            self.final_norm = math.sqrt(nom)
            if(self.parameters["print_level"] >= 0):
                print self.reason[self.reasonid]
                print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
            return
        
        # start iteration
        self.iter = 1
        while True:
            alpha = nom/den
            x.axpy(alpha,self.d)        # x = x + alpha d
            self.r.axpy(-alpha, self.z) # r = r - alpha A d
            
            self.B.solve(self.z, self.r)     # z = B^-1 r
            betanom = self.r.inner(self.z)
            
            if self.parameters["print_level"] == 1:
                print " Iteration : ", self.iter, " (B r, r) = ", betanom
                
            if betanom < r0:
                self.converged = True
                self.reasonid = 1
                self.final_norm = math.sqrt(betanom)
                if(self.parameters["print_level"] >= 0):
                    print self.reason[self.reasonid]
                    print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
                break
            
            self.iter += 1
            if self.iter > self.parameters["max_iter"]:
                self.converged = False
                self.reasonid = 0
                self.final_norm = math.sqrt(betanom)
                if(self.parameters["print_level"] >= 0):
                    print self.reason[self.reasonid]
                    print "Not Converged. Final residual norm ", self.final_norm
                break
            
            beta = betanom/nom
            self.d *= beta
            self.d.axpy(1., self.z)  #d = z + beta d
            
            self.A.mult(self.d,self.z)   # z = A d
            
            den = self.d.inner(self.z)
            
            if den <= 0.0:
                self.converged = True
                self.reasonid = 2
                self.final_norm = math.sqrt(nom)
                if(self.parameters["print_level"] >= 0):
                    print self.reason[self.reasonid]
                    print "Converged in ", self.iter, " iterations with final norm ", self.final_norm
                break
            
            nom = betanom


                
            
            
        
        
