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

import unittest 
import dolfin as dl
import ufl
import numpy as np

import sys
sys.path.append('../../')
from hippylib import *

class J_op:
    """
    J_op implements action of BA^-1C. A s.p.d.
    """
    def __init__(self, B, Asolver, C):
        self.Asolver = Asolver
        self.B = B
        self.C = C
        self.temp0 = dl.Vector(self.C.mpi_comm())
        self.temp1 = dl.Vector(self.C.mpi_comm())
        self.temp1help = dl.Vector(self.C.mpi_comm())
        self.B.init_vector(self.temp0,0)
        self.B.init_vector(self.temp1,1)
        self.B.init_vector(self.temp1help,1)
        
    def init_vector(self, x, dim):
        if dim == 1:
            self.C.init_vector(x, 1)
        elif dim == 0:
            self.B.init_vector(x,0)
        else:
            assert(dim in [0,1])

    def mpi_comm(self):
        return self.B.mpi_comm()

    def mult(self, x, y):
        self.C.mult(x,self.temp1)
        self.Asolver.solve(self.temp1help, self.temp1)
        self.B.mult(self.temp1help,y)
        
    def transpmult(self, x, y):
        self.B.transpmult(x,self.temp1)
        self.Asolver.solve(self.temp1help, self.temp1)
        self.C.transpmult(self.temp1help, y)
        
        
class TestRandomizedSVD(unittest.TestCase):
    def setUp(self):
        mesh = dl.UnitSquareMesh(10, 10)
        self.mpi_rank = dl.MPI.rank(mesh.mpi_comm())
        self.mpi_size = dl.MPI.size(mesh.mpi_comm())
        
        Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)

        uh,vh = dl.TrialFunction(Vh1),dl.TestFunction(Vh1)
        mh = dl.TrialFunction(Vh1)
        
        # Define B
        ndim = 2
        ntargets = 10
        np.random.seed(seed=1)
        targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
        B = assemblePointwiseObservation(Vh1, targets)

        # Define Asolver
        alpha = dl.Constant(0.1)
        varfA = ufl.inner(ufl.grad(uh), ufl.grad(vh))*ufl.dx +\
                    alpha*ufl.inner(uh,vh)*ufl.dx
        A = dl.assemble(varfA)
        Asolver = PETScKrylovSolver(A.mpi_comm(), "cg", amg_method())
        Asolver.set_operator(A)
        Asolver.parameters["maximum_iterations"] = 100
        Asolver.parameters["relative_tolerance"] = 1e-12

        # Define M
        varfC = ufl.inner(mh,vh)*ufl.dx
        C = dl.assemble(varfC)

        self.J = J_op(B,Asolver,C)
        
        self.k_evec = 10
        p_evec = 50
        
        myRandom = Random(self.mpi_rank, self.mpi_size)

        x_vec = dl.Vector(C.mpi_comm())
        C.init_vector(x_vec,1)

        self.Omega = MultiVector(x_vec,self.k_evec+p_evec)
        myRandom.normal(1.,self.Omega)
        
        y_vec = dl.Vector(C.mpi_comm())
        B.init_vector(y_vec,0)
        self.Omega_adj = MultiVector(y_vec,self.k_evec+p_evec)
        myRandom.normal(1.,self.Omega_adj)


    def testAccuracyEnhancedSVD(self):
        
        self.U,self.sigma,self.V = accuracyEnhancedSVD(self.J,self.Omega,self.k_evec,s=1)
        assert np.all(self.sigma>0)
        
        UtU = self.U.dot_mv(self.U)
        err = UtU - np.eye(UtU.shape[0], dtype=UtU.dtype)
        err_Uortho = np.linalg.norm(err, 'fro')

        VtV = self.V.dot_mv(self.V)
        err = VtV - np.eye(VtV.shape[0], dtype=VtV.dtype)
        err_Vortho = np.linalg.norm(err, 'fro')
            
        nvec  = self.U.nvec()
        AV = MultiVector(self.U[0], nvec)
        MatMvMult(self.J, self.V, AV)
        UtAV = np.diag(AV.dot_mv(self.U))
        r_1 = np.zeros_like(self.sigma)

        AtU = MultiVector(self.V[0], nvec)
        MatMvTranspmult(self.J, self.U, AtU)
        VtAtU = np.diag(AtU.dot_mv(self.V))
        r_2 = np.zeros_like(self.sigma)

        for i,sigma_i in enumerate(self.sigma):
            r_1[i] = np.abs(UtAV[i] - sigma_i)
            r_2[i] = np.abs(VtAtU[i] - sigma_i)

        assert err_Uortho < 1e-8
        assert err_Vortho < 1e-8
        assert np.all(r_1 < np.maximum( 5e-2,0.1*self.sigma))
        assert np.all(r_2 < np.maximum( 5e-2,0.1*self.sigma))
            
    def testSinglePassSVD(self):
        # Only check execution at this time
        U, sigma, V = singlePassSVD(self.J,self.Omega,self.Omega_adj,self.k_evec)

        

if __name__ == '__main__':
    unittest.main()
