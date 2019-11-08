# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019, The University of Texas at Austin 
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

from hippylib import *
import dolfin as dl
import numpy as np

class Jacobian:
    """
    This class implements matrix free application of the Jacobian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    
    Type :code:`help(modelTemplate)` for more information on which methods model should implement.
    """
    def __init__(self, model):
        """
        Construct the Jacobian operator
        """
        self.model = model
        self.gauss_newton_approx = True

        self.ncalls = 0
        
        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(ADJOINT)
        self.rhs_adj2 = model.generate_vector(ADJOINT)
        self.uhat    = model.generate_vector(STATE)
        self.phat    = model.generate_vector(ADJOINT)
        self.yhelp = model.generate_vector(PARAMETER)

        self.Bu = dl.Vector()
        self.model.misfit.B.init_vector(self.Bu,0)

        self.Ctphat = model.generate_vector(PARAMETER)

        self.shape = (self.Bu.get_local().shape[0],self.rhs_fwd.get_local().shape[0])
    
    def init_vector(self, x, dim):
        """
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

        - :code:`x`: the vector to reshape.
        - :code:`dim`: if 0 then :code:`x` will be made compatible with the range of the Jacobian, if 1 then :code:`x` will be made compatible with the domain of the Jacobian.
               
         """
        if dim == 0:
            self.model.misfit.B.init_vector(x,0)
        elif dim == 1:
            self.model.init_parameter(x)
        else: 
            raise

        
        
    def mult(self,x,y):
        """
        Apply the Jacobian :code:`x`. Return the result in :code:`y`.
        """
        self.J(x,y)

        self.ncalls += 1

    def transpmult(self,x,y):
        """
        Apply the Jacobian :code:`x`. Return the result in :code:`y`.
        """
        self.JT(x,y)

        self.ncalls += 1

    def J(self,x,y):
        """
        Apply the Jacobian
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        assert hasattr(self.model.misfit,'B'), 'Misfit must have attribute B'
        self.model.misfit.B.mult(self.uhat,y)

        y *= (1./np.sqrt(self.model.misfit.noise_variance))
        

    def JT(self,x,y):
        """
        Apply the transpose of the Jacobian      
        """
        assert hasattr(self.model.misfit,'B'), 'Misfit must have attribute B'
        self.model.misfit.B.transpmult(x,self.rhs_adj)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyCt(self.phat, y)

        y *= (1./np.sqrt(self.model.misfit.noise_variance))
        
            
    def JTJ(self,x,y):
        """
        Apply the Gauss-Newton approximation of the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.        
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.misfit.B.mult(self.uhat,self.Bu)
        self.model.misfit.B.transpmult(self.Bu,self.rhs_adj)
        self.rhs_adj *= (1./self.model.misfit.noise_variance)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyCt(self.phat, y)


    def JJT(self,x,y):
        """
        Apply the inside out Gauss-Newton approximation of the reduced Hessian JJT to the vector :code:`x`.
        Return the result in :code:`y`.        
        """
        self.model.misfit.B.transpmult(x,self.rhs_adj)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyCt(self.phat, self.Ctphat)
        self.model.applyC(self.Ctphat, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.misfit.B.mult(self.uhat,y)
        y *= (1./self.model.misfit.noise_variance)

        

    
