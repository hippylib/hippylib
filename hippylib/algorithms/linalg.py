# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
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

import dolfin as dl
from petsc4py import PETSc

from ..utils.random import parRandom
import numpy as np

def amg_method(amg_type="ml_amg"):
    """
    Determine which AMG preconditioner to use.
    If available use the preconditioner suggested by the user (ML is default).
    If not available use  petsc_amg.
    """
    for pp in dl.krylov_solver_preconditioners():
        if pp[0] == amg_type:
            return amg_type
        
    return 'petsc_amg'

def MatMatMult(A,B):
    """
    Compute the matrix-matrix product :math:`AB`.
    """
    Amat = dl.as_backend_type(A).mat()
    Bmat = dl.as_backend_type(B).mat()
    out = Amat.matMult(Bmat)
    rmap, _ = Amat.getLGMap()
    _, cmap = Bmat.getLGMap()
    out.setLGMap(rmap, cmap)
    return dl.Matrix(dl.PETScMatrix(out))

def MatPtAP(A,P):
    """
    Compute the triple matrix product :math:`P^T A P`.
    """
    Amat = dl.as_backend_type(A).mat()
    Pmat = dl.as_backend_type(P).mat()
    out = Amat.PtAP(Pmat, fill=1.0)
    _, out_map = Pmat.getLGMap()
    out.setLGMap(out_map, out_map)
    return dl.Matrix(dl.PETScMatrix(out))

def MatAtB(A,B):
    """
    Compute the matrix-matrix product :math:`A^T B`.
    """
    Amat = dl.as_backend_type(A).mat()
    Bmat = dl.as_backend_type(B).mat()
    out = Amat.transposeMatMult(Bmat)
    _, rmap = Amat.getLGMap()
    _, cmap = Bmat.getLGMap()
    out.setLGMap(rmap, cmap)
    return dl.Matrix(dl.PETScMatrix(out))

def Transpose(A):
    """
    Compute the matrix transpose
    """
    Amat = dl.as_backend_type(A).mat()
    AT = PETSc.Mat()
    Amat.transpose(AT)
    rmap, cmap = Amat.getLGMap()
    AT.setLGMap(cmap, rmap)
    return dl.Matrix( dl.PETScMatrix(AT) )

def SetToOwnedGid(v, gid, val):
    v[gid] = val

    
def GetFromOwnedGid(v, gid):
    return v[gid]
    

def to_dense(A, mpi_comm = dl.MPI.comm_world ):
    """
    Convert a sparse matrix A to dense.
    For debugging only.
    """
    v = dl.Vector(mpi_comm)
    A.init_vector(v)
    nprocs = dl.MPI.size(mpi_comm)
    
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
        x = dl.Vector(mpi_comm)
        Ax = dl.Vector(mpi_comm)
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


def trace(A, mpi_comm = dl.MPI.comm_world ):
    """
    Compute the trace of a sparse matrix :math:`A`.
    """
    v = dl.Vector(mpi_comm)
    A.init_vector(v)
    nprocs = dl.MPI.size(mpi_comm)
    
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
    ej, xj = dl.Vector(d.mpi_comm()), dl.Vector(d.mpi_comm())
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
    x, b = dl.Vector(d.mpi_comm()), dl.Vector(d.mpi_comm())
    
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
    def __init__(self,S,mpi_comm=dl.MPI.comm_world, init_vector = None):
        self.S = S
        self.tmp = dl.Vector(mpi_comm)
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
    def __init__(self,op, mpi_comm=dl.MPI.comm_world):
        self.op = op
        self.tmp = dl.Vector(mpi_comm)
        
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
