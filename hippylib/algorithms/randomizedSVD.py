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
    print('Y_shape = ',MV_shape(Y))
    print('Z shape = ',(Z[0].get_local().shape[0],Z.nvec()))
    MatMvMult(A,Omega,Y)


    for i in range(s):
        MatMvTranspmult(A,Y,Z)
        MatMvMult(A, Z, Y)

    Q = MultiVector(Y)
    Q.orthogonalize()
    print('Q shape = ',(Q[0].get_local().shape[0],Q.nvec()))


    BT = MultiVector(Omega)
    MatMvTranspmult(A,Q,BT)

    BT_numpy = MV_to_dense(BT)

    R_B = BT.orthogonalize()


    print('||R|| = ', np.linalg.norm(R_B))
    print('The shape of R is ',R_B.shape)

    B = BT_numpy.T
    print('B shape = ',B.shape)
    print('||B|| = ',np.linalg.norm(B))

    U_hat,d,V = np.linalg.svd(B,full_matrices = False) 

    print('U_hat shape = ', U_hat.shape)
    print('d shape = ', d.shape)
    print('V shape = ', V.shape)

    U_hat = U_hat[:,:k]
    d = d[:k]
    V = V[:,:k]

    print('U_hat shape = ', U_hat.shape)
    print('d shape = ', d.shape)
    print('V shape = ', V.shape)

    U = MultiVector(y_vec, k)   

    print('U shape = ',(U[0].get_local().shape[0],U.nvec()))
    print('Q shape = ',(Q[0].get_local().shape[0],Q.nvec()))
    print('U_hat shape = ', U_hat.shape)

    MvDSmatMult(Q, U_hat, U)
        
    return U, d, V


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
  