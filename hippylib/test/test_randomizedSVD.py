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
    def __init__(self, Asolver, M):
        self.Asolver = Asolver
        self.M = M
        self.temp = dl.Vector()
        self.M.init_vector(self.temp,0)
        
    def init_vector(self, x, dim):
        self.M.init_vector(x,dim)

    def mpi_comm(self):
        return self.M.mpi_comm()

    def mult(self, x, y):
        self.Asolver.solve(y, self.M*x)

    def transpmult(self, x, y):
        self.Asolver.solve(self.temp,x)
        self.M.transpmult(self.temp,y)


class TestRandomizedSVD(unittest.TestCase):
    def setUp(self):
        mesh = dl.UnitSquareMesh(10, 10)
        self.rank = dl.MPI.rank(mesh.mpi_comm())
        Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
        Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)

        uh,vh = dl.TrialFunction(Vh2),dl.TestFunction(Vh2)
        mh = dl.TrialFunction(Vh1)

        alpha = 1.0

        varfA = dl.inner(dl.nabla_grad(uh), dl.nabla_grad(vh))*dl.dx +\
                    alpha*dl.inner(uh,vh)*dl.dx

        A = dl.assemble(varfA)

        varfM = dl.inner(mh,vh)*dl.dx
        M = dl.assemble(varfM)

        Asolver = dl.PETScKrylovSolver(A.mpi_comm(), "cg", amg_method())
        Asolver.set_operator(A)
        Asolver.parameters["maximum_iterations"] = 100
        Asolver.parameters["relative_tolerance"] = 1e-12

        self.J = J_op(Asolver,M)
        myRandom = Random()

        x_vec = dl.Vector()
        M.init_vector(x_vec,1)

        k_evec = 10
        p_evec = 50
        Omega = MultiVector(x_vec,k_evec+p_evec)

        myRandom.normal(1.,Omega)

        self.U,self.d,self.V = accuracyEnhancedSVD(self.J,Omega,k_evec,s=2)
        
        
    def testOrthogonalityU(self):
        UtU = self.U.dot_mv(self.U)
        err = UtU - np.eye(UtU.shape[0], dtype=UtU.dtype)
        err_Uortho = np.linalg.norm(err, 'fro')

        if self.rank == 0:
            assert err_Uortho < 1e-8

    def testOrthogonalityV(self):
        VtV = self.V.dot_mv(self.V)
        err = VtV - np.eye(VtV.shape[0], dtype=VtV.dtype)
        err_Vortho = np.linalg.norm(err, 'fro')
        if self.rank == 0:
            assert err_Vortho < 1e-8

    def testResidualUtAV(self):
        nvec  = self.U.nvec()
        AV = MultiVector(self.U[0], nvec)
        MatMvMult(self.J, self.V, AV)
        UtAV = np.diag(AV.dot_mv(self.U))
        r_1 = np.zeros_like(self.d)
        for i,d_i in enumerate(self.d):
            r_1[i] = min(np.abs(UtAV[i] + d_i),np.abs(UtAV[i] - d_i))
        if self.rank == 0:
            assert np.all(r_1 < 5e-2)

    def testResidualVtAtU(self):
        nvec  = self.V.nvec()
        AtU = MultiVector(self.V[0], nvec)
        MatMvTranspmult(self.J, self.U, AtU)
        VtAtU = np.diag(AtU.dot_mv(self.V))
        r_2 = np.zeros_like(self.d)
        for i,d_i in enumerate(self.d):
            r_2[i] = min(np.abs(VtAtU[i] + d_i),np.abs(VtAtU[i] - d_i))
        if self.rank == 0:
            assert np.all(r_2 < 5e-2)
        

if __name__ == '__main__':
    unittest.main()