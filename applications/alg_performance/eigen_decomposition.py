'''
Created on Jun 3, 2020

@author: uvilla
'''

import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *



def singlePassG_original(A, B, Binv, Omega,k):
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
    Q = MultiVector(Omega)
    MatMvMult(A, Omega, Ybar)
    MatMvMult(Solver2Operator(Binv), Ybar, Q)
    
    BQ, _ = Q.Borthogonalize(B)
    
    X = Omega.dot_mv(BQ)
    W = Omega.dot_mv(Ybar)
    XTinvW = np.linalg.solve(np.transpose(X),W)
    T = np.linalg.solve(X, np.transpose(XTinvW))
                        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    
        
    return d, U

class Hop:
    """
    Implements the action of CtA^-1BtBA-1C with A s.p.d.
    """
    def __init__(self,B, Asolver, C):

        self.B = B
        self.Asolver = Asolver
        self.C = C

        self.temp = dl.Vector(self.C.mpi_comm())
        self.temphelp = dl.Vector(self.C.mpi_comm())
        self.C.init_vector(self.temp,0)
        self.C.init_vector(self.temphelp,0)
        
        self.Bhelp = dl.Vector(self.C.mpi_comm())
        self.B.init_vector(self.Bhelp,0)


    def mult(self,x,y):
        self.C.mult(x, self.temp)
        self.Asolver.solve(self.temphelp,self.temp)
        self.B.mult(self.temphelp,self.Bhelp)
        self.B.transpmult(self.Bhelp,self.temp)
        self.Asolver.solve(self.temphelp,self.temp)
        self.C.transpmult(self.temphelp,y)
        
    def mpi_comm(self):
        return self.C.mpi_comm()

    def init_vector(self,x,dim):
        self.C.init_vector(x,1)


if __name__ == "__main__":
    mesh = dl.UnitSquareMesh(10, 10)
    mpi_rank = dl.MPI.rank(mesh.mpi_comm())
    mpi_size = dl.MPI.size(mesh.mpi_comm())
    
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)

    uh,vh = dl.TrialFunction(Vh1),dl.TestFunction(Vh1)
    mh,test_mh = dl.TrialFunction(Vh1),dl.TestFunction(Vh1)
    
    ## Set up B
    ndim = 2
    ntargets = 200
    np.random.seed(seed=1)
    targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    B = assemblePointwiseObservation(Vh1, targets)

    ## Set up Asolver
    eps   = dl.Constant(0.1)
    alpha = dl.Constant(1.0)
    varfA = eps*ufl.inner(ufl.grad(uh), ufl.grad(vh))*ufl.dx +\
                alpha*ufl.inner(uh,vh)*ufl.dx
    A = dl.assemble(varfA)
    Asolver = PETScKrylovSolver(A.mpi_comm(), "cg", amg_method())
    Asolver.set_operator(A)
    Asolver.parameters["maximum_iterations"] = 100
    Asolver.parameters["relative_tolerance"] = 1e-12

    ## Set up C
    varfC = ufl.inner(mh,vh)*ufl.dx
    C = dl.assemble(varfC)

    Hop = Hop(B, Asolver, C)

    ## Set up RHS Matrix M.
    varfM = ufl.inner(mh,test_mh)*ufl.dx
    M = dl.assemble(varfM)
    Minv = PETScKrylovSolver(M.mpi_comm(), "cg", "jacobi")
    Minv.set_operator(M)
    Minv.parameters["maximum_iterations"] = 100
    Minv.parameters["relative_tolerance"] = 1e-12

    myRandom = Random(mpi_rank, mpi_size)

    x_vec = dl.Vector(mesh.mpi_comm())
    Hop.init_vector(x_vec,1)

    k_evec = 50
    p_evec = 20
    Omega_ref = MultiVector(x_vec,k_evec+p_evec)
    myRandom.normal(1., Omega_ref)
    k_evec = k_evec
    
    d_dp,U_true = doublePassG(Hop,M,Minv,MultiVector(Omega_ref),k_evec,s=1)
    d_our, U_our  = singlePassG(Hop,M,Minv,MultiVector(Omega_ref),k_evec,s=1)
    d_their, U_their = singlePassG_original(Hop,M,Minv,MultiVector(Omega_ref),k_evec)
    
    if mpi_rank==0:
        plt.semilogy(d_dp, '*-b', label='Double Pass')
        plt.semilogy(d_our, '*g', label='Single Pass (Our)')
        plt.semilogy(d_their, '*r', label='Single Pass (SaibabaEtAl)')
        plt.legend()
        plt.show()
    
        
    