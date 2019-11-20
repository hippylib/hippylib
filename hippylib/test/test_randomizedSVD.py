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

import unittest 
import dolfin as dl
import numpy as np

from numpy.testing import assert_allclose

import sys
sys.path.append('../../')
from hippylib import *

class J_op:
    """
    J_op implements action of BA^-1M. A s.p.d.
    """
    def __init__(self, B, Asolver, M):
        self.Asolver = Asolver
        self.B = B
        self.M = M 
        self.temp0 = dl.Vector(self.M.mpi_comm())
        self.temp1 = dl.Vector(self.M.mpi_comm())
        self.temp1help = dl.Vector(self.M.mpi_comm())
        self.B.init_vector(self.temp0,0)
        self.B.init_vector(self.temp1,1)
        self.B.init_vector(self.temp1help,1)
        
    def init_vector(self, x, dim):
        if dim == 1:
            self.M.init_vector(x, 1)
        elif dim == 0:
            self.B.init_vector(x,0)
        else:
            assert(dim in [0,1])

    def mpi_comm(self):
        return self.B.mpi_comm()

    def mult(self, x, y):
        self.M.mult(x,self.temp1)
        self.Asolver.solve(self.temp1help, self.temp1)
        self.B.mult(self.temp1help,y)
        
    def transpmult(self, x, y):
        self.B.transpmult(x,self.temp1)
        self.Asolver.solve(self.temp1help, self.temp1)
        self.M.transpmult(self.temp1help, y)
        
        
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
        B = assemblePointwiseStateObservation(Vh1, targets)

        # Define Asolver
        alpha = dl.Constant(0.1)
        varfA = dl.inner(dl.nabla_grad(uh), dl.nabla_grad(vh))*dl.dx +\
                    alpha*dl.inner(uh,vh)*dl.dx
        A = dl.assemble(varfA)
        Asolver = dl.PETScKrylovSolver(A.mpi_comm(), "cg", amg_method())
        Asolver.set_operator(A)
        Asolver.parameters["maximum_iterations"] = 100
        Asolver.parameters["relative_tolerance"] = 1e-12

        # Define M
        varfM = dl.inner(mh,vh)*dl.dx
        M = dl.assemble(varfM)

        self.J = J_op(B,Asolver,M)
        myRandom = Random(self.mpi_rank, self.mpi_size)

        x_vec = dl.Vector(M.mpi_comm())
        M.init_vector(x_vec,1)

        k_evec = 10
        p_evec = 50
        Omega = MultiVector(x_vec,k_evec+p_evec)

        myRandom.normal(1.,Omega)
        # The way that this spectrum clusters power iteration makes the algorithm worse
        # Bringing the orthogonalization inside of the power iteration could fix this
        self.U,self.d,self.V = accuracyEnhancedSVD(self.J,Omega,k_evec,s=1)
        assert np.all(self.d>0)

    def testAccuracyEnhancedSVD(self):
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
        r_1 = np.zeros_like(self.d)

        AtU = MultiVector(self.V[0], nvec)
        MatMvTranspmult(self.J, self.U, AtU)
        VtAtU = np.diag(AtU.dot_mv(self.V))
        r_2 = np.zeros_like(self.d)

        for i,d_i in enumerate(self.d):
            r_1[i] = min(np.abs(UtAV[i] + d_i),np.abs(UtAV[i] - d_i))
            r_2[i] = min(np.abs(VtAtU[i] + d_i),np.abs(VtAtU[i] - d_i))

        if self.rank == 0:
            assert err_Uortho < 1e-8
            assert err_Vortho < 1e-8
            assert np.all(r_1 < np.maximum( 5e-2,0.1*self.d))
            assert np.all(r_2 < np.maximum( 5e-2,0.1*self.d))

        

if __name__ == '__main__':
    unittest.main()
