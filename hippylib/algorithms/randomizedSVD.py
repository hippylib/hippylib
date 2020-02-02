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
from .multivector import MultiVector, MatMvMult, MatMvTranspmult, MvDSmatMult
from ..utils import experimental
import numpy as np
from scipy.linalg import solve_sylvester
"""
Randomized algorithms for the solution of singular value decomposition problem (SVD)

REFERENCES:

Nathan Halko, Per Gunnar Martinsson, and Joel A. Tropp,
Finding structure with randomness:
Probabilistic algorithms for constructing approximate matrix decompositions,
SIAM Review, 53 (2011), pp. 217-288.

Per Gunnar Martinsson
Randomized Methods for Matrix Computations
https://arxiv.org/pdf/1607.01649v3.pdf
"""

def accuracyEnhancedSVD(A,Omega,k,s=1,check=False):
    """
    The accuracy enhanced randomized singular value decomposition from  [2].
    
    Inputs:

    - :code:`A`: the m x n rectangular operator for which we need to estimate the dominant left-right singular vector / value triplets.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    
    Outputs:
    - :code:`U`: the estimate of the :math:`k` dominant left singular vectors of of :math:`A,\\, U^T U = I_k`.
    - :code:`sigma`: the estimate of the :math:`k` dominant singular values of :math:`A`.
    - :code:`V`: the estimate of the :math:`k` dominant right singular vectors of :math:`A,\\, V^T V = I_k`.
    
    """

    # Check compatibility of operator A
    assert hasattr(A,'transpmult'), 'Operator A must have member function transpmult'

    nvec  = Omega.nvec()
    assert nvec >= k, 'Omega must have at least k columns' 

    Z = MultiVector(Omega)

    y_vec = dl.Vector(A.mpi_comm())
    A.init_vector(y_vec,0)
    Y = MultiVector(y_vec,nvec)
    MatMvMult(A,Omega,Y)
    # Perform power iteration for the range approximation step
    for i in range(s):
        MatMvTranspmult(A,Y,Z)
        MatMvMult(A, Z, Y)
    # First orthogonal matrix for left singular vectors
    # Note: Bringing the orthogonalization inside of the power iteration could improve accuracy
    Q = MultiVector(Y)
    Q.orthogonalize()


    # Form BT = A^TQ (B = Q^TA) and orthogonalize in one step
    # This becomes the orthogonal matrix for right singular vectors
    Q_Bt = MultiVector(Omega)
    MatMvTranspmult(A,Q,Q_Bt)
    R_Bt = Q_Bt.orthogonalize()

    V_hat,sigma,U_hat = np.linalg.svd(R_Bt,full_matrices = False) 

    # Select the first k columns
    U_hat = U_hat[:,:k]
    sigma = sigma[:k]
    V_hat = V_hat[:,:k]

    U = MultiVector(y_vec, k)
    MvDSmatMult(Q, U_hat, U)   
    V = MultiVector(Omega[0],k)
    MvDSmatMult(Q_Bt, V_hat, V)

    if check:
        check_SVD(A,U,sigma,V)

    return U, sigma, V

@experimental(name = 'singlePassSVD',version='3.0.0', msg='Accuracy of these computations cannot be guaranteed.')
def singlePassSVD(A,Omega_c,Omega_r,k,check=False):
    """
    The single pass randomized singular value decomposition from  [2].
    
    Inputs:

    - :code:`A`: the m x n rectangular operator for which we need to estimate the dominant left-right singular vector / value triplets.
    - :code:`Omega_c`: an n x (k +p) random gassian matrix with :math:`n \\geq k` columns.
    - :code:`Omega_r`: an m x (k +p) random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    
    Outputs:
    - :code:`U`: the estimate of the :math:`k` dominant left singular vectors of of :math:`A,\\, U^T U = I_k`.
    - :code:`sigma`: the estimate of the :math:`k` dominant singular values of :math:`A`.
    - :code:`V`: the estimate of the :math:`k` dominant right singular vectors of :math:`A,\\, V^T V = I_k`.
    
    """

    # Check compatibility of operator A
    assert hasattr(A,'transpmult'), 'Operator A must have member function transpmult'

    nvec  = Omega_c.nvec()
    assert nvec >= k, 'Omega_c must have at least k columns' 
    assert Omega_r.nvec() == nvec, 'Omega_c and Omega_r must have the same number of columns' 

    Y_c = MultiVector(Omega_r)
    Y_r = MultiVector(Omega_c)

    MatMvMult(A,Omega_c,Y_c)
    MatMvTranspmult(A,Omega_r,Y_r)

    # Orthogonalize 
    Q_c = MultiVector(Y_c)
    Q_c.orthogonalize()
    Q_r = MultiVector(Y_r)
    Q_r.orthogonalize()

    # Need to solve the system of equations for C: (Omega_rT Q_c)C = Y_rT Q_r
    #                                              C(Q_rT Omega_c) = Q_cTY_c
    Omega_rTQ_c = Q_c.dot_mv(Omega_r)
    Y_rTQ_r = Q_r.dot_mv(Y_r)
    Q_rTOmega_c = Omega_c.dot_mv(Q_r)
    Q_cTY_c = Y_c.dot_mv(Q_c)
    # Sylvester solution to the least squares problem
    sylvester_lead = Omega_rTQ_c.T@Omega_rTQ_c
    sylvester_trail = Q_rTOmega_c@(Q_rTOmega_c.T)
    sylvester_rhs = ((Omega_rTQ_c.T)@Y_rTQ_r) + (Q_cTY_c@(Q_rTOmega_c.T))
    C = solve_sylvester(sylvester_lead,sylvester_trail,sylvester_rhs)
    sylvester_error = np.linalg.norm(sylvester_lead@C + C@sylvester_trail - sylvester_rhs)
    assert sylvester_error < 1e-4, 'Issue with sylvester solver'

    U_hat,sigma,V_hat = np.linalg.svd(C,full_matrices = False) 

    # Select the first k columns
    U_hat = U_hat[:,:k]
    sigma = sigma[:k]
    V_hat = V_hat[:,:k]

    U = MultiVector(Omega_r[0], k)
    MvDSmatMult(Q_c, U_hat, U)   
    V = MultiVector(Omega_c[0],k)
    MvDSmatMult(Q_r, V_hat, V)

    if check:
        check_SVD(A,U,sigma,V)

    return U, sigma, V


def check_SVD(A, U, sigma,V,tol = 1e-1):
    """
    Test the frobenious norm of  :math:`U^TU - I_k`.

    Test the frobenious norm of  :math:`(V^TV) - I_k`.
    
    Test the :math:`l_2` norm of the residual: :math:`r_1[i] = U[i]^T A V[i] - sigma[i]`.

    Test the :math:`l_2` norm of the residual: :math:`r_2[i] = V[i]^TA^T U[i] -  sigma[i]`.
    """
    assert U.nvec() == V.nvec(), "U and V second dimension need to agree"

    nvec  = U.nvec()
    AV = MultiVector(U[0], nvec)
    MatMvMult(A, V, AV)
    AtU = MultiVector(V[0],nvec)
    MatMvTranspmult(A,U,AtU)

    # # Residual checks
    UtAV = np.diag(AV.dot_mv(U))
    r_1 = np.zeros_like(sigma)
    for i,sigma_i in enumerate(sigma):
        r_1[i] = np.abs(UtAV[i] - sigma_i)
    # r_1 = Ut_AV - d
    
    VtAtU = np.diag(AtU.dot_mv(V))
    # r_2 = VtAtU - d
    r_2 = np.zeros_like(sigma)
    for i,sigma_i in enumerate(sigma):
        r_2[i] = np.abs(VtAtU[i] - sigma_i)
    
    # Orthogonality checks check
    UtU = U.dot_mv(U)
    err = UtU - np.eye(nvec, dtype=UtU.dtype)
    err_Uortho = np.linalg.norm(err, 'fro')
    
    VtV = U.dot_mv(U)
    err = VtV - np.eye(nvec, dtype=VtV.dtype)
    err_Vortho = np.linalg.norm(err, 'fro')
    
    mpi_comm = U[0].mpi_comm()
    rank = dl.MPI.rank(mpi_comm)
    if rank == 0:
        print( "|| UtU - I ||_F = ", err_Uortho)
        print( "|| VtV - I ||_F = ", err_Vortho)
        print( "|utAv - sigma| < tol ",np.all(r_1 < tol))
        print(r_1)
        print( "|vtAtu - sigma| < tol ",np.all(r_2 < tol))
        print(r_2)
  
