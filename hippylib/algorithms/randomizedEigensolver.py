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

from dolfin import Vector, MPI
from .linalg import Solver2Operator
from .multivector import MultiVector, MatMvMult, MvDSmatMult
import numpy as np

"""
Randomized algorithms for the solution of Hermitian Eigenvalues Problems (HEP)
and Generalized Hermitian Eigenvalues Problems (GHEP).

In particular we provide an implementation of the single and double pass algorithms
and some convergence test.

REFERENCES:

Nathan Halko, Per Gunnar Martinsson, and Joel A. Tropp,
Finding structure with randomness:
Probabilistic algorithms for constructing approximate matrix decompositions,
SIAM Review, 53 (2011), pp. 217-288.

Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis,
Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application
to computing Karhunen-Loeve expansion,
Numerical Linear Algebra with Applications, to appear.
"""

def singlePass(A,Omega,k,s=1,check=False):
    """
    The single pass algorithm for the Hermitian Eigenvalues Problems (HEP) as presented in [1].
    
    Inputs:

    - :code:`A`: the operator for which we need to estimate the dominant eigenpairs.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    
    Outputs:

    - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
    - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T U = I_k`.
    """
    nvec  = Omega.nvec()
    assert(nvec >= k )
    
    Y_pr = MultiVector(Omega)
    Y = MultiVector(Omega)
    for i in range(s):
        Y_pr.swap(Y)
        MatMvMult(A, Y_pr, Y)
        
    Q = MultiVector(Y)
    Q.orthogonalize()
        
    Zt = Y_pr.dot_mv(Q)
    Wt = Y.dot_mv(Q)
        
    Tt = np.linalg.solve(Zt, Wt)
                
    T = .5*Tt + .5*Tt.T
        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = MultiVector(Omega[0], k)   
    MvDSmatMult(Q, V, U)
    
    if check:
        check_std(A, U, d)
        
    return d, U

def doublePass(A,Omega,k,s,check = False):
    """
    The double pass algorithm for the HEP as presented in [1].
    
    Inputs:

    - :code:`A`: the operator for which we need to estimate the dominant eigenpairs.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    - :code:`s`: the number of power iterations for selecting the subspace.
    
    Outputs:

    - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
    - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T U = I_k`.
    """    
        
    nvec  = Omega.nvec()
    assert(nvec >= k )
    
    Q = MultiVector(Omega)
    Y = MultiVector(Omega[0], nvec)
    for i in range(s):
        MatMvMult(A, Q, Y)
        Q.swap(Y)
    Q.orthogonalize()
    
    AQ = MultiVector(Omega[0], nvec)
    MatMvMult(A, Q, AQ)
                
    T = AQ.dot_mv(Q)
        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
     
    U = MultiVector(Omega[0], k)   
    MvDSmatMult(Q, V, U)
    
    if check:
        check_std(A, U, d)
        
    return d, U

def singlePassG(A, B, Binv, Omega,k, s = 1, check = False):
    """
    The single pass algorithm for the Generalized Hermitian Eigenvalues Problems (GHEP) as presented in [2].
    
    Inputs:

    - :code:`A`: the operator for which we need to estimate the dominant generalized eigenpairs.
    - :code:`B`: the right-hand side operator.
    - :code:`Binv`: the inverse of the right-hand side operator.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    - :code:`s`: the number of power iterations for selecting the subspace.
    
    Outputs:

    - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
    - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T B U = I_k`.
    """
    nvec  = Omega.nvec()
    
    assert(nvec >= k )
    
    Ybar = MultiVector(Omega[0], nvec)
    Y_pr = MultiVector(Omega)
    Q = MultiVector(Omega)
    for i in range(s):
        Y_pr.swap(Q)
        MatMvMult(A, Y_pr, Ybar)
        MatMvMult(Solver2Operator(Binv), Ybar, Q)
    
    BQ, _ = Q.Borthogonalize(B)
    
    Xt = Y_pr.dot_mv(BQ)
    Wt = Ybar.dot_mv(Q)
    Tt = np.linalg.solve(Xt,Wt)
                
    T = .5*Tt + .5*Tt.T
        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    
    if check:
        check_g(A,B, U, d)
        
    return d, U

def doublePassG(A, B, Binv, Omega, k, s = 1, check = False):
    """
    The double pass algorithm for the GHEP as presented in [2].
    
    Inputs:

    - :code:`A`: the operator for which we need to estimate the dominant generalized eigenpairs.
    - :code:`B`: the right-hand side operator.
    - :code:`Binv`: the inverse of the right-hand side operator.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    - :code:`s`: the number of power iterations for selecting the subspace.
    
    Outputs:

    - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
    - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T B U = I_k`.
    """        
    nvec  = Omega.nvec()
    
    assert(nvec >= k )
    
    Ybar = MultiVector(Omega[0], nvec)
    Q = MultiVector(Omega)
    for i in range(s):
        MatMvMult(A, Q, Ybar)
        MatMvMult(Solver2Operator(Binv), Ybar, Q)
    
    Q.Borthogonalize(B)
    AQ = MultiVector(Omega[0], nvec)
    MatMvMult(A, Q, AQ)
    
    T = AQ.dot_mv(Q)
                        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    
    if check:
        check_g(A,B, U, d)
            
    return d, U

def check_std(A, U, d):
    """
    Test the frobenious norm of  :math:`U^TU - I_k`.

    Test the frobenious norm of  :math:`(V^TAV) - I_k`, with :math:`V = U D^{-1/2}`.
    
    Test the :math:`l_2` norm of the residual: :math:`r[i] = A U[i] - d[i] U[i]`.
    """
    nvec  = U.nvec()
    AU = MultiVector(U[0], nvec)
    MatMvMult(A, U, AU)
    
    # Residual checks
    diff = MultiVector(AU)
    diff.axpy(-d, U)
    res_norms = diff.norm("l2")
    
    # B-ortho check
    UtU = U.dot_mv(U)
    err = UtU - np.eye(nvec, dtype=UtU.dtype)
    err_Bortho = np.linalg.norm(err, 'fro')
    
    #A-ortho check
    V = MultiVector(U)
    scaling = np.power(np.abs(d), -0.5)
    V.scale(scaling)
    AU.scale(scaling)
    VtAV = AU.dot_mv(V)
    err = VtAV - np.diag(np.sign(d))#np.eye(nvec, dtype=VtAV.dtype)
    err_Aortho = np.linalg.norm(err, 'fro')
    
    mpi_comm = U[0].mpi_comm()
    rank = MPI.rank(mpi_comm)
    if rank == 0:
        print( "|| UtU - I ||_F = ", err_Bortho)
        print( "|| VtAV - I ||_F = ", err_Aortho, " with V = U D^{-1/2}")
        print( "lambda", "||Au - lambda*u||_2")
        for i in range(res_norms.shape[0]):
            print( "{0:5e} {1:5e}".format(d[i], res_norms[i]))


def check_g(A,B, U, d):
    """
    Test the frobenious norm of  :math:`U^TBU - I_k`.

    Test the frobenious norm of  :math:`(V^TAV) - I_k`, with :math:`V = U D^{-1/2}`.

    Test the :math:`l_2` norm of the residual: :math:`r[i] = A U[i] - d[i] B U[i]`.
    """
    nvec  = U.nvec()
    AU = MultiVector(U[0], nvec)
    BU = MultiVector(U[0], nvec)
    MatMvMult(A, U, AU)
    MatMvMult(B, U, BU)
    
    # Residual checks
    diff = MultiVector(AU)
    diff.axpy(-d, BU)
    res_norms = diff.norm("l2")
    
    # B-ortho check
    UtBU = BU.dot_mv(U)
    err = UtBU - np.eye(nvec, dtype=UtBU.dtype)
    err_Bortho = np.linalg.norm(err, 'fro')
    
    #A-ortho check
    V = MultiVector(U)
    scaling = np.power(np.abs(d), -0.5)
    V.scale(scaling)
    AU.scale(scaling)
    VtAV = AU.dot_mv(V)
    err = VtAV - np.eye(nvec, dtype=VtAV.dtype)
    err_Aortho = np.linalg.norm(err, 'fro')
    
    mpi_comm = U[0].mpi_comm()
    rank = MPI.rank(mpi_comm)
    if rank == 0:
        print( "|| UtBU - I ||_F = ", err_Bortho)
        print( "|| VtAV - I ||_F = ", err_Aortho, " with V = U D^{-1/2}")
        print( "lambda", "||Au - lambdaBu||_2")
        for i in range(res_norms.shape[0]):
            print( "{0:5e} {1:5e}".format(d[i], res_norms[i]) )    