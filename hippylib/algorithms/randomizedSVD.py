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

from dolfin import Vector, MPI
from .linalg import Solver2Operator
from .multivector import MultiVector, MatMvMult, MatMvTranspmult, MvDSmatMult
import numpy as np

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

# These functions are useful for debugging
def MV_to_dense(multivector):
    multivector_shape = (multivector[0].get_local().shape[0],multivector.nvec())
    as_np_array = np.zeros(multivector_shape)
    for i in range(multivector_shape[-1]):
        temp = multivector[i].get_local()
        # print('For iteration i ||get_local|| = ', np.linalg.norm(temp))
        as_np_array[:,i] = temp
    return as_np_array

def MV_shape(multivector):
    return (multivector[0].get_local().shape[0],multivector.nvec())



def accuracyEnhancedSVD(A,Omega,k,s=1,check=False):
    """
    The accuracy enhanced randomized singular value decomposition from  [2].
    
    Inputs:

    - :code:`A`: the rectangular operator for which we need to estimate the dominant left-right singular vector pairs.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    
    Outputs:
    - :code:`U`: the estimate of the :math:`k` dominant left singular vectors of of :math:`A,\\, U^T U = I_k`.
    - :code:`d`: the estimate of the :math:`k` dominant singular values of :math:`A`.
    - :code:`V`: the estimate of the :math:`k` dominant right singular vectors of :math:`A,\\, V^T V = I_k`.
    
    """

    # Check compatibility of operator A
    assert hasattr(A,'transpmult'), 'Operator A must have member function transpmult'

    nvec  = Omega.nvec()
    assert(nvec >= k )

    Z = MultiVector(Omega)

    y_vec = Vector()
    A.init_vector(y_vec,0)
    Y = MultiVector(y_vec,nvec)
    MatMvMult(A,Omega,Y)
    # Perform power iteration for the range approximation step
    for i in range(s):
        MatMvTranspmult(A,Y,Z)
        MatMvMult(A, Z, Y)
    # First orthogonal matrix for left singular vectors
    Q = MultiVector(Y)
    Q.orthogonalize()


    # Form BT = A^TQ (B = Q^TA) and orthogonalize in one step
    # This becomes the orthogonal matrix for right singular vectors
    Q_Bt = MultiVector(Omega)
    MatMvTranspmult(A,Q,Q_Bt)
    R_Bt = Q_Bt.orthogonalize()

    V_hat,d,U_hat = np.linalg.svd(R_Bt,full_matrices = False) 

    # Select the first k columns
    U_hat = U_hat[:,:k]
    d = d[:k]
    V_hat = V_hat[:,:k]

    U = MultiVector(y_vec, k)
    MvDSmatMult(Q, U_hat, U)   
    V = MultiVector(Omega[0],k)
    MvDSmatMult(Q_Bt, V_hat, V)

    if check:
        check_SVD(A,U,d,V)

    return U, d, V


def check_SVD(A, U, d,V,tol = 1e-1):
    """
    Test the frobenious norm of  :math:`U^TU - I_k`.

    Test the frobenious norm of  :math:`(V^TV) - I_k`.
    
    Test the :math:`l_2` norm of the residual: :math:`r_1[i] = A V[i] -  U[i] d[i]`.

    Test the :math:`l_2` norm of the residual: :math:`r_2[i] = A^T U[i] -  V[i] d[i]`.
    """
    assert U.nvec() == V.nvec(), "U and V second dimension need to agree"

    nvec  = U.nvec()
    AV = MultiVector(U[0], nvec)
    MatMvMult(A, V, AV)
    AtU = MultiVector(V[0],nvec)
    MatMvTranspmult(A,U,AtU)

    # # Residual checks
    Ut_AV = np.diag(AV.dot_mv(U))
    r_1 = np.zeros_like(d)
    for i,d_i in enumerate(d):
        r_1[i] = min(np.abs(Ut_AV[i] + d_i),np.abs(Ut_AV[i] - d_i))
    # r_1 = Ut_AV - d
    
    VtAtU = np.diag(AtU.dot_mv(V))
    # r_2 = VtAtU - d
    r_2 = np.zeros_like(d)
    for i,d_i in enumerate(d):
        r_2[i] = min(np.abs(VtAtU[i] + d_i),np.abs(VtAtU[i] - d_i))
    
    # Orthogonality checks check
    UtU = U.dot_mv(U)
    err = UtU - np.eye(nvec, dtype=UtU.dtype)
    err_Uortho = np.linalg.norm(err, 'fro')
    
    VtV = U.dot_mv(U)
    err = VtV - np.eye(nvec, dtype=VtV.dtype)
    err_Vortho = np.linalg.norm(err, 'fro')
    
    mpi_comm = U[0].mpi_comm()
    rank = MPI.rank(mpi_comm)
    if rank == 0:
        print( "|| UtU - I ||_F = ", err_Uortho)
        print( "|| VtV - I ||_F = ", err_Vortho)
        print( "|utAv - sigma| < tol ",np.all(r_1 < tol))
        print( "|vtAtu - sigma| < tol ",np.all(r_2 < tol))

  