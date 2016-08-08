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
import math
import matplotlib.pyplot as plt

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

def singlePass(A,Omega,k):
    """
    The single pass algorithm for the HEP as presented in [1].
    Inputs:
    - A: the operator for which we need to estimate the dominant eigenpairs.
    - Omega: a random gassian matrix with m >= k columns.
    - k: the number of eigenpairs to extract.
    
    Outputs:
    - d: the estimate of the k dominant eigenvalues of A
    - U: the estimate of the k dominant eigenvectors of A. U^T U = I_k
    """
    w = Vector()
    y = Vector()
    A.init_vector(w,1)
    A.init_vector(y,0)
    
    nvec  = Omega.shape[1]
    
    assert(nvec >= k )
    
    Y = np.zeros(Omega.shape)
    
    for ivect in range(0,nvec):
        w.set_local(Omega[:,ivect])
        A.mult(w,y)
        Y[:,ivect] = y.array()
                
    Q,_ = np.linalg.qr(Y)
        
    Zt = np.dot(Omega.T, Q)
    Wt = np.dot(Y.T, Q)
        
    Tt = np.linalg.solve(Zt, Wt)
                
    T = .5*Tt + .5*Tt.T
        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = np.dot(Q,V)
        
    return d, U

def doublePass(A,Omega,k):
    """
    The double pass algorithm for the HEP as presented in [1].
    Inputs:
    - A: the operator for which we need to estimate the dominant eigenpairs.
    - Omega: a random gassian matrix with m >= k columns.
    - k: the number of eigenpairs to extract.
    
    Outputs:
    - d: the estimate of the k dominant eigenvalues of A
    - U: the estimate of the k dominant eigenvectors of A. U^T U = I_k
    """
    w = Vector()
    y = Vector()
    A.init_vector(w,1)
    A.init_vector(y,0)
    
    nvec  = Omega.shape[1]
    
    assert(nvec >= k )
    
    Y = np.zeros(Omega.shape)
    
    for ivect in range(0,nvec):
        w.set_local(Omega[:,ivect])
        A.mult(w,y)
        Y[:,ivect] = y.array()
                
    Q,_ = np.linalg.qr(Y)
    
    AQ = np.zeros(Omega.shape)
    for ivect in range(0,nvec):
        w.set_local(Q[:,ivect])
        A.mult(w,y)
        AQ[:,ivect] = y.array()
                
    T = np.dot(Q.T, AQ)
        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = np.dot(Q,V)
        
    return d, U

def singlePassG(A, B, Binv, Omega,k, check_Bortho = False, check_Aortho=False, check_residual = False):
    """
    The single pass algorithm for the GHEP as presented in [2].
    B-orthogonalization is achieved using the PreCholQR algorithm.
    
    Inputs:
    - A: the operator for which we need to estimate the dominant generalized eigenpairs.
    - B: the rhs operator
    - Omega: a random gassian matrix with m >= k columns.
    - k: the number of eigenpairs to extract.
    
    Outputs:
    - d: the estimate of the k dominant eigenvalues of A
    - U: the estimate of the k dominant eigenvectors of A. U^T B U = I_k
    """
    w = Vector()
    ybar = Vector()
    y = Vector()
    A.init_vector(w,1)
    A.init_vector(y,0)
    A.init_vector(ybar,0)
    
    nvec  = Omega.shape[1]
    
    assert(nvec >= k )
    
    Ybar = np.zeros(Omega.shape)
    Y = np.zeros(Omega.shape)
    
    for ivect in range(0,nvec):
        w.set_local(Omega[:,ivect])
        A.mult(w,ybar)
        Binv.solve(y, ybar)
        Ybar[:,ivect] = ybar.array()
        Y[:,ivect] = y.array()
                
    Z,_ = np.linalg.qr(Y)
    BZ = np.zeros(Omega.shape)
    for ivect in range(0,nvec):
        w.set_local(Z[:,ivect])
        B.mult(w,y)
        BZ[:, ivect] = y
        
    R = np.linalg.cholesky( np.dot(Z.T,BZ ))
    Q = np.linalg.solve(R, Z.T).T
    BQ = np.linalg.solve(R, BZ.T).T
    
    Xt = np.dot(Omega.T, BQ)
    Wt = np.dot(Ybar.T, Q)
    Tt = np.linalg.solve(Xt,Wt)
                
    T = .5*Tt + .5*Tt.T
        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = np.dot(Q,V)
    
    if check_Bortho:
        BorthogonalityTest(B, U)
    if check_Aortho: 
        AorthogonalityCheck(A, U, d)
    if check_residual:
        residualCheck(A,B, U, d)
        
    return d, U

def doublePassG(A, B, Binv, Omega,k, check_Bortho = False, check_Aortho=False, check_residual = False):
    """
    The double pass algorithm for the GHEP as presented in [2].
    B-orthogonalization is achieved using the PreCholQR algorithm.
    
    Inputs:
    - A: the operator for which we need to estimate the dominant generalized eigenpairs.
    - B: the rhs operator
    - Omega: a random gassian matrix with m >= k columns.
    - k: the number of eigenpairs to extract.
    
    Outputs:
    - d: the estimate of the k dominant eigenvalues of A
    - U: the estimate of the k dominant eigenvectors of A. U^T B U = I_k
    """
    w = Vector()
    ybar = Vector()
    y = Vector()
    A.init_vector(w,1)
    A.init_vector(y,0)
    A.init_vector(ybar,0)
    
    nvec  = Omega.shape[1]
    
    assert(nvec >= k )
    
    Ybar = np.zeros(Omega.shape)
    Y = np.zeros(Omega.shape)
    
    for ivect in range(0,nvec):
        w.set_local(Omega[:,ivect])
        A.mult(w,ybar)
        Binv.solve(y, ybar)
        Ybar[:,ivect] = ybar.array()
        Y[:,ivect] = y.array()
                
    Z,_ = np.linalg.qr(Y)
    BZ = np.zeros(Omega.shape)
    for ivect in range(0,nvec):
        w.set_local(Z[:,ivect])
        B.mult(w,y)
        BZ[:, ivect] = y
        
    R = np.linalg.cholesky( np.dot(Z.T,BZ ))
    Q = np.linalg.solve(R, Z.T).T
    
    AQ = np.zeros(Q.shape, dtype=Q.dtype)
    for ivect in range(0,nvec):
        w.set_local(Q[:,ivect])
        A.mult(w,ybar)
        AQ[:,ivect] = ybar.array()
                
    T = np.dot(Q.T, AQ)
        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = np.dot(Q,V)
    
    if check_Bortho:
        BorthogonalityTest(B, U)
    if check_Aortho: 
        AorthogonalityCheck(A, U, d)
    if check_residual:
        residualCheck(A,B, U, d)
        
    return d, U

            

def BorthogonalityTest(B, U):
    """
    Test the frobenious norm of  U^TBU - I_k
    """
    BU = np.zeros(U.shape)
    Bu = Vector()
    u = Vector()
    B.init_vector(Bu,0)
    B.init_vector(u,1)
    
    nvec  = U.shape[1]
    for i in range(0,nvec):
        u.set_local(U[:,i])
        B.mult(u,Bu)
        BU[:,i] = Bu.array()
        
    UtBU = np.dot(U.T, BU)
    err = UtBU - np.eye(nvec, dtype=UtBU.dtype)
    print "|| UtBU - I ||_F = ", np.linalg.norm(err, 'fro')
    
def AorthogonalityCheck(A, U, d):
    """
    Test the frobenious norm of  D^{-1}(U^TAU) - I_k
    """
    V = np.zeros(U.shape)
    AV = np.zeros(U.shape)
    Av = Vector()
    v = Vector()
    A.init_vector(Av,0)
    A.init_vector(v,1)
    
    nvec  = U.shape[1]
    for i in range(0,nvec):
        v.set_local(U[:,i])
        v *= 1./math.sqrt(d[i])
        A.mult(v,Av)
        AV[:,i] = Av.array()
        V[:,i] = v.array()
        
    VtAV = np.dot(V.T, AV)    
    err = VtAV - np.eye(nvec, dtype=VtAV.dtype)
    
#    plt.imshow(np.abs(err))
#    plt.colorbar()
#    plt.show()
    
    print "i, ||Vt(i,:)AV(:,i) - I_i||_F, V[:,i] = 1/sqrt(lambda_i) U[:,i]"
    for i in range(1,nvec+1):
        print i, np.linalg.norm(err[0:i,0:i], 'fro')

    
def residualCheck(A,B, U, d):
    """
    Test the l2 norm of the residual:
    r[:,i] = d[i] B U[:,i] - A U[:,i]
    """
    u = Vector()
    Au = Vector()
    Bu = Vector()
    Binv_r = Vector()
    A.init_vector(u,1)
    A.init_vector(Au, 0)
    B.init_vector(Bu, 0)
    B.init_vector(Binv_r, 0)
    
    nvec = d.shape[0]
    
    print "lambda", "||Au - lambdaBu||"
    for i in range(0,nvec):
        u.set_local(U[:,i])
        A.mult(u,Au)
        B.mult(u,Bu)
        Au.axpy(-d[i], Bu)
        print d[i], Au.norm("l2")
        