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

from dolfin import compile_extension_module, Vector, PETScKrylovSolver, MPI, la_index_dtype, mpi_comm_world
from ..utils.random import parRandom
import os
import numpy as np

def amg_method(amg_type="ml_amg"):
    """
    Determine which AMG preconditioner to use.
    If available use the preconditioner suggested by the user (ML is default).
    If not available use  petsc_amg.
    """
    S = PETScKrylovSolver()
    for pp in S.preconditioners():
        if pp[0] == amg_type:
            return amg_type
        
    return 'petsc_amg'

abspath = os.path.dirname( os.path.abspath(__file__) )
source_directory = os.path.join(abspath,"cpp_linalg")
header_file = open(os.path.join(source_directory,"linalg.h"), "r")
code = header_file.read()
header_file.close()
cpp_sources = ["linalg.cpp"]  

include_dirs = [".", source_directory]
for ss in ['PROFILE_INSTALL_DIR', 'PETSC_DIR', 'SLEPC_DIR']:
    if ss in os.environ.keys():
        include_dirs.append(os.environ[ss]+'/include')
        
cpp_module = compile_extension_module(
                code=code, source_directory=source_directory,
                sources=cpp_sources, include_dirs=include_dirs)

def MatMatMult(A,B):
    """
    Compute the matrix-matrix product :math:`AB`.
    """
    s = cpp_module.cpp_linalg()
    return s.MatMatMult(A,B)

def MatPtAP(A,P):
    """
    Compute the triple matrix product :math:`P^T A P`.
    """
    s = cpp_module.cpp_linalg()
    return s.MatPtAP(A,P)

def MatAtB(A,B):
    """
    Compute the matrix-matrix product :math:`A^T B`.
    """
    s = cpp_module.cpp_linalg()
    return s.MatAtB(A,B)

def Transpose(A):
    """
    Compute the matrix transpose
    """
    s = cpp_module.cpp_linalg()
    return s.Transpose(A)

def SetToOwnedGid(v, gid, val):
    s = cpp_module.cpp_linalg()
    s.SetToOwnedGid(v, gid, val)
    
def GetFromOwnedGid(v, gid):
    s = cpp_module.cpp_linalg()
    return s.GetFromOwnedGid(v, gid)
    

def to_dense(A, mpi_comm = mpi_comm_world() ):
    """
    Convert a sparse matrix A to dense.
    For debugging only.
    """
    v = Vector(mpi_comm)
    A.init_vector(v)
    nprocs = MPI.size(mpi_comm)
    
    if nprocs > 1:
        raise Exception("to_dense is only serial")
    
    if hasattr(A, "getrow"):
        n  = A.size(0)
        m  = A.size(1)
        B = np.zeros( (n,m), dtype=np.float64)
        for i in range(0,n):
            [j, val] = A.getrow(i)
            B[i,j] = val
        
        return B
    else:
        x = Vector(mpi_comm)
        Ax = Vector(mpi_comm)
        A.init_vector(x,1)
        A.init_vector(Ax,0)
        
        n = Ax.get_local().shape[0]
        m = x.get_local().shape[0]
        B = np.zeros( (n,m), dtype=np.float64) 
        for i in range(0,m):
            i_ind = np.array([i], dtype=np.intc)
            x.set_local(np.ones(i_ind.shape), i_ind)
            x.apply("sum_values")
            A.mult(x,Ax)
            B[:,i] = Ax.get_local()
            x.set_local(np.zeros(i_ind.shape), i_ind)
            x.apply("sum_values")
            
        return B


def trace(A, mpi_comm = mpi_comm_world() ):
    """
    Compute the trace of a sparse matrix :math:`A`.
    """
    v = Vector(mpi_comm)
    A.init_vector(v)
    nprocs = MPI.size(mpi_comm)
    
    if nprocs > 1:
        raise Exception("trace is only serial")
    
    n  = A.size(0)
    tr = 0.
    for i in range(0,n):
        [j, val] = A.getrow(i)
        tr += val[j == i]
    return tr

def get_diagonal(A, d):
    """
    Compute the diagonal of the square operator :math:`A`.
    Use :code:`Solver2Operator` if :math:`A^{-1}` is needed.
    """
    ej, xj = Vector(d.mpi_comm()), Vector(d.mpi_comm())
    A.init_vector(ej,1)
    A.init_vector(xj,0)
                    
    g_size = ej.size()    
    d.zero()
    for gid in range(g_size):
        owns_gid = ej.owns_index(gid)
        if owns_gid:
            SetToOwnedGid(ej, gid, 1.)
        ej.apply("insert")
        A.mult(ej,xj)
        if owns_gid:
            val = GetFromOwnedGid(xj, gid)
            SetToOwnedGid(d, gid, val)
            SetToOwnedGid(ej, gid, 0.)
        ej.apply("insert")
        
    d.apply("insert")

    

def estimate_diagonal_inv2(Asolver, k, d):
    """
    An unbiased stochastic estimator for the diagonal of :math:`A^{-1}`.
    :math:`d = [ \sum_{j=1}^k v_j .* A^{-1} v_j ] ./ [ \sum_{j=1}^k v_j .* v_j ]`
    where

    - :math:`v_j` are i.i.d. :math:`\mathcal{N}(0, I)`
    - :math:`.*` and :math:`./` represent the element-wise multiplication and division
      of vectors, respectively.
      
    Reference:
        `Costas Bekas, Effrosyni Kokiopoulou, and Yousef Saad, 
        An estimator for the diagonal of a matrix, 
        Applied Numerical Mathematics, 57 (2007), pp. 1214-1229.`
    """
    x, b = Vector(d.mpi_comm()), Vector(d.mpi_comm())
    
    if hasattr(Asolver, "init_vector"):
        Asolver.init_vector(x,1)
        Asolver.init_vector(b,0)
    else:       
        Asolver.get_operator().init_vector(x,1)
        Asolver.get_operator().init_vector(b,0)
        
    d.zero()
    for i in range(k):
        x.zero()
        parRandom.normal(1., b)
        Asolver.solve(x,b)
        x *= b
        d.axpy(1./float(k), x)
        
class DiagonalOperator:
    def __init__(self, d):
        self.d = d
        
    def init_vector(self,x,dim):
        x.init(self.d.local_range())
        
    def mult(self,x,y):
        tmp = self.d*x
        y.zero()
        y.axpy(1., x)
        
    def inner(self,x,y):
        tmp = self.d*y
        return x.inner(tmp)
    
class Solver2Operator:
    def __init__(self,S,mpi_comm=mpi_comm_world(), init_vector = None):
        self.S = S
        self.tmp = Vector(mpi_comm)
        self.my_init_vector = init_vector
        
        if self.my_init_vector is None:
            if hasattr(self.S, "init_vector"):
                self.my_init_vector = self.S.init_vector
            elif hasattr(self.S, "operator"):
                self.my_init_vector = self.S.operator().init_vector
            elif hasattr(self.S, "get_operator"):
                self.my_init_vector = self.S.get_operator().init_vector
        
    def init_vector(self, x, dim):
        if self.my_init_vector:
            self.my_init_vector(x,dim)
        else:
            raise NotImplementedError("Solver2Operator.init_vector")
        
        
    def mult(self,x,y):
        self.S.solve(y,x)
        
    def inner(self, x, y):
        self.S.solve(self.tmp,y)
        return self.tmp.inner(x)
    
class Operator2Solver:
    def __init__(self,op, mpi_comm=mpi_comm_world()):
        self.op = op
        self.tmp = Vector(mpi_comm)
        
    def init_vector(self, x, dim):
        if hasattr(self.op, "init_vector"):
            self.op.init_vector(x,dim)
        else:
            raise
        
    def solve(self,y,x):
        self.op.mult(x,y)
        
    def inner(self, x, y):
        self.op.mult(y,self.tmp)
        return self.tmp.inner(x)
