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
import numpy as np

class LowRankOperator:
    """
    This class model the action of a low rank operator A = U D U^T.
    Here D is a diagonal matrix, and the columns of are orthonormal
    in some weighted inner-product.
    """
    def __init__(self,d,U, my_init_vector = None):
        """
        Construct the low rank operator given d and U.
        """
        self.d = d
        self.U = U
        self.my_init_vector = my_init_vector
        
    def init_vector(self, x, dim):
        """
        Initialize x to be compatible with the range/domain of A.
        """
        assert self.my_init_vector is not None
        self.my_init_vector(x,dim)
        
    def mult(self,x,y):
        """
        Compute y = Ax = U D U^T x
        """
        Utx = np.dot( self.U.T, x.array() )
        dUtx = self.d*Utx
        y.set_local(np.dot(self.U, dUtx))
        
    def inner(self, x, y):
        Utx = np.dot( self.U.T, x.array() )
        Uty = np.dot( self.U.T, y.array() )
        return np.sum(self.d*Utx*Uty)
        
        
    def solve(self, sol, rhs):
        """
        Compute sol = U D^-1 U^T x
        """
        Utx = np.dot( self.U.T, rhs.array() )
        dinvUtx = Utx / self.d
        sol.set_local(np.dot(self.U, dinvUtx))
        
    def get_diagonal(self, diag):
        """
        Compute the diagonal of A.
        """
        V = self.U * self.d
        diag.set_local(np.sum(V*self.U, 1))
        
    def trace(self,W=None):
        """
        Compute the trace of A.
        If the weight W is given compute the trace of W^1/2AW^1/2.
        This is equivalent to
        tr_W(A) = \sum_i lambda_i,
        where lambda_i are the generalized eigenvalues of
        A x = lambda W^-1 x.
        
        Note if U is a W-orthogonal matrix then
        tr_W(A) = \sum_i D(i,i). 
        """
        if W is None:
            diagUtU = np.sum(self.U*self.U,0)
            tr = np.sum(self.d*diagUtU)
        else:
            WU = np.zeros(self.U.shape, dtype=self.U.dtype)
            u, wu = Vector(), Vector()
            W.init_vector(u,1)
            W.init_vector(wu,0)
            for i in range(self.U.shape[1]):
                u.set_local(self.U[:,i])
                W.mult(u,wu)
                WU[:,i] = wu.array()
            diagWUtU = np.sum(WU*self.U,0)
            tr = np.sum(self.d*diagWUtU)
            
        return tr
    
    def trace2(self,W=None):
        """
        Compute the trace of A*A (Note this is the square of Frob norm, since A is symmetic).
        If the weight W is provided, it will compute the trace of (AW)^2.
        
        This is equivalent to 
        tr_W(A) = \sum_i lambda_i^2,
        where lambda_i are the generalized eigenvalues of
        A x = lambda W^-1 x.
        
        Note if U is a W-orthogonal matrix then
        tr_W(A) = \sum_i D(i,i)^2. 
        """
        if W is None:
            UtU = np.dot(self.U.T, self.U)
            dUtU = self.d[:,None] * UtU #diag(d)*UtU.
            tr2 = np.sum(dUtU*dUtU)
        else:
            WU = np.zeros(self.U.shape, dtype=self.U.dtype)
            u, wu = Vector(), Vector()
            W.init_vector(u,1)
            W.init_vector(wu,0)
            for i in range(self.U.shape[1]):
                u.set_local(self.U[:,i])
                W.mult(u,wu)
                WU[:,i] = wu.array()
            UtWU = np.dot(self.U.T, WU)
            dUtWU = self.d[:,None] * UtWU #diag(d)*UtU.
            tr2 = np.power(np.linalg.norm(dUtWU),2)
            
        return tr2