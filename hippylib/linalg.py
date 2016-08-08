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

from dolfin import compile_extension_module, Vector, PETScKrylovSolver, Function
import os
import numpy as np

def amg_method():
    """
    Determine which AMG preconditioner to use.
    If avaialable use ML, which is faster than the PETSc one.
    """
    S = PETScKrylovSolver()
    for pp in S.preconditioners():
        if pp[0] == 'ml_amg':
            return 'ml_amg'
        
    return 'petsc_amg'

abspath = os.path.dirname( os.path.abspath(__file__) )
sdir = os.path.join(abspath,"cpp_linalg")
header_file = open(os.path.join(sdir,"linalg.h"), "r")
code = header_file.read()
header_file.close()
cpp_sources = ["linalg.cpp"]  
cpp_module = compile_extension_module(
code=code, source_directory=sdir, sources=cpp_sources,
include_dirs=[".",  sdir])

def MatMatMult(A,B):
    """
    Compute the matrix-matrix product A*B.
    """
    s = cpp_module.cpp_linalg()
    return s.MatMatMult(A,B)

def MatPtAP(A,P):
    """
    Compute the triple matrix product P^T*A*P.
    """
    s = cpp_module.cpp_linalg()
    return s.MatPtAP(A,P)

def MatAtB(A,B):
    """
    Compute the matrix-matrix product A^T*B.
    """
    s = cpp_module.cpp_linalg()
    return s.MatAtB(A,B)

def Transpose(A):
    """
    Compute the matrix transpose
    """
    s = cpp_module.cpp_linalg()
    return s.Transpose(A)
    

def to_dense(A):
    """
    Convert a sparse matrix A to dense.
    For debugging only.
    """
    if hasattr(A, "getrow"):
        n  = A.size(0)
        m  = A.size(1)
        B = np.zeros( (n,m), dtype=np.float64)
        for i in range(0,n):
            [j, val] = A.getrow(i)
            B[i,j] = val
        
        return B
    else:
        x = Vector()
        Ax = Vector()
        A.init_vector(x,1)
        A.init_vector(Ax,0)
        
        n = Ax.array().shape[0]
        m = x.array().shape[0]
        B = np.zeros( (n,m), dtype=np.float64) 
        for i in range(0,m):
            i_ind = np.array([i], dtype=np.intc)
            x.set_local(np.ones(i_ind.shape), i_ind)
            A.mult(x,Ax)
            B[:,i] = Ax.array()
            x.set_local(np.zeros(i_ind.shape), i_ind)
            
        return B


def trace(A):
    """
    Compute the trace of a sparse matrix A.
    """
    n  = A.size(0)
    tr = 0.
    for i in range(0,n):
        [j, val] = A.getrow(i)
        tr += val[j == i]
    return tr

def get_diagonal(A, d, solve_mode=True):
    """
    Compute the diagonal of the square operator A
    or its inverse A^{-1} (if solve_mode=True).
    """
    ej, xj = Vector(), Vector()

    if hasattr(A, "init_vector"):
        A.init_vector(ej,1)
        A.init_vector(xj,0)
    else:       
        A.get_operator().init_vector(ej,1)
        A.get_operator().init_vector(xj,0)
        
    ncol = ej.size()
    da = np.zeros(ncol, dtype=ej.array().dtype)
    
    for j in range(ncol):
        ej[j] = 1.
        if solve_mode:
            A.solve(xj, ej)
        else:
            A.mult(ej,xj)
        da[j] = xj[j]
        ej[j] = 0.
        
    d.set_local(da)

      


def estimate_diagonal_inv2(Asolver, k, d):
    """
    An unbiased stochastic estimator for the diagonal of A^-1.
    d = [ \sum_{j=1}^k vj .* A^{-1} vj ] ./ [ \sum_{j=1}^k vj .* vj ]
    where
    - vj are i.i.d. ~ N(0, I)
    - .* and ./ represent the element-wise multiplication and division
      of vectors, respectively.
      
    REFERENCE:
    Costas Bekas, Effrosyni Kokiopoulou, and Yousef Saad,
    An estimator for the diagonal of a matrix,
    Applied Numerical Mathematics, 57 (2007), pp. 1214-1229.
    """
    x, b = Vector(), Vector()
    
    if hasattr(Asolver, "init_vector"):
        Asolver.init_vector(x,1)
        Asolver.init_vector(b,0)
    else:       
        Asolver.get_operator().init_vector(x,1)
        Asolver.get_operator().init_vector(b,0)
    
    num = np.zeros(b.array().shape, dtype = b.array().dtype)
    den = np.zeros(num.shape, dtype = num.dtype)
    for i in range(k):
        x.zero()
        b.set_local(np.random.randn(num.shape[0]))
        Asolver.solve(x,b)
        num = num +  ( x.array() * b.array() )
        den = den +  ( b.array() * b.array() )
        
    d.set_local( num / den )
        
def randn_perturb(x, std_dev):
    """
    Add a Gaussian random perturbation to x:
    x = x + eta, eta ~ N(0, std_dev^2 I)
    """
    n = x.array().shape[0]
    noise = np.random.normal(0, 1, n)
    x.set_local(x.array() + std_dev*noise)
    
class Solver2Operator:
    def __init__(self,S):
        self.S = S
        self.tmp = Vector()
        
    def init_vector(self, x, dim):
        if hasattr(self.S, "init_vector"):
            self.S.init_vector(x,dim)
        elif hasattr(self.S, "operator"):
            self.S.operator().init_vector(x,dim)
        else:
            raise
        
    def mult(self,x,y):
        self.S.solve(y,x)
        
    def inner(self, x, y):
        self.S.solve(self.tmp,y)
        return self.tmp.inner(x)
    
def vector2Function(x,Vh, **kwargs):
    """
    Wrap a finite element vector x into a finite element function in the space Vh.
    kwargs is optional keywords arguments to be passed to the construction of a dolfin Function
    """
    fun = Function(Vh,**kwargs)
    fun.vector().zero()
    fun.vector().axpy(1., x)
    
    return fun