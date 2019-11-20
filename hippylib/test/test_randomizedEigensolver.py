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
from hippylib import (MultiVector, Random, assemblePointwiseObservation, amg_method,
                      singlePass, singlePassG, doublePass, doublePassG, MatMvMult)
                      


class Aop:
    """
    Implements the action of MtA^-1BtBA-1M with A s.p.d.
    """
    def __init__(self,B, Asolver, M):

        self.B = B
        self.Asolver = Asolver
        self.M = M

        self.temp = dl.Vector(M.mpi_comm())
        self.temphelp = dl.Vector(M.mpi_comm())
        self.M.init_vector(self.temp,0)
        self.M.init_vector(self.temphelp,0)
        self.Bhelp = dl.Vector(M.mpi_comm())
        self.B.init_vector(self.Bhelp,0)


    def mult(self,x,y):
        self.M.mult(x, self.temp)
        self.Asolver.solve(self.temphelp,self.temp)
        self.B.mult(self.temphelp,self.Bhelp)
        self.B.transpmult(self.Bhelp,self.temp)
        self.Asolver.solve(self.temphelp,self.temp)
        self.M.transpmult(self.temphelp,y)
        
    def mpi_comm(self):
        return self.M.mpi_comm()

    def init_vector(self,x,dim):
        self.M.init_vector(x,1)



class TestRandomizedEigensolver(unittest.TestCase):
    def setUp(self):
        mesh = dl.UnitSquareMesh(10, 10)
        self.mpi_rank = dl.MPI.rank(mesh.mpi_comm())
        self.mpi_size = dl.MPI.size(mesh.mpi_comm())
        
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
        alpha = dl.Constant(1.0)
        varfA = dl.inner(dl.grad(uh), dl.grad(vh))*dl.dx +\
                    alpha*dl.inner(uh,vh)*dl.dx
        A = dl.assemble(varfA)
        Asolver = dl.PETScKrylovSolver(A.mpi_comm(), "cg", amg_method())
        Asolver.set_operator(A)
        Asolver.parameters["maximum_iterations"] = 100
        Asolver.parameters["relative_tolerance"] = 1e-12

        ## Set up M
        varfM = dl.inner(mh,vh)*dl.dx
        M = dl.assemble(varfM)

        self.Aop = Aop(B, Asolver, M)

        varfG = dl.inner(mh,test_mh)*dl.dx
        self.rhs_G = dl.assemble(varfG)
        self.rhs_Ginv = dl.PETScKrylovSolver(self.rhs_G.mpi_comm(), "cg", amg_method())
        self.rhs_Ginv.set_operator(self.rhs_G)
        self.rhs_Ginv.parameters["maximum_iterations"] = 100
        self.rhs_Ginv.parameters["relative_tolerance"] = 1e-12

        myRandom = Random(self.mpi_rank, self.mpi_size)

        x_vec = dl.Vector(mesh.mpi_comm())
        self.Aop.init_vector(x_vec,1)

        k_evec = 10
        p_evec = 50
        self.Omega = MultiVector(x_vec,k_evec+p_evec)
        self.k_evec = k_evec

        myRandom.normal(1.,self.Omega)

        
    def testSinglePass(self):
        d,U = singlePass(self.Aop,self.Omega,self.k_evec,s=2)
        nvec  = U.nvec()
        AU = MultiVector(U[0], nvec)
        MatMvMult(self.Aop, U, AU)

        # Residual checks
        diff = MultiVector(AU)
        diff.axpy(-d, U)
        res_norms = diff.norm("l2")
        # B-orthogonality
        UtU = U.dot_mv(U)
        err = UtU - np.eye(UtU.shape[0], dtype=UtU.dtype)
        err_Bortho = np.linalg.norm(err, 'fro')
        # A-orthogonality
        V = MultiVector(U)
        scaling = np.power(np.abs(d), -0.5)
        V.scale(scaling)
        AU.scale(scaling)
        VtAV = AU.dot_mv(V)
        err = VtAV - np.diag(np.sign(d))
        err_Aortho = np.linalg.norm(err, 'fro')

        if self.mpi_rank == 0:
            assert err_Bortho < 1e-8
            assert err_Aortho < 1e-4
            assert np.all(res_norms < 1e-4)

    def testDoublePass(self):
        d,U = doublePass(self.Aop,self.Omega,self.k_evec,s=2)
        nvec  = U.nvec()
        AU = MultiVector(U[0], nvec)
        MatMvMult(self.Aop, U, AU)

        # Residual checks
        diff = MultiVector(AU)
        diff.axpy(-d, U)
        res_norms = diff.norm("l2")
        # B-orthogonality
        UtU = U.dot_mv(U)
        err = UtU - np.eye(UtU.shape[0], dtype=UtU.dtype)
        err_Bortho = np.linalg.norm(err, 'fro')
        # A-orthogonality
        V = MultiVector(U)
        scaling = np.power(np.abs(d), -0.5)
        V.scale(scaling)
        AU.scale(scaling)
        VtAV = AU.dot_mv(V)
        err = VtAV - np.diag(np.sign(d))
        err_Aortho = np.linalg.norm(err, 'fro')

        if self.mpi_rank == 0:
            assert err_Bortho < 1e-8
            assert err_Aortho < 1e-4
            assert np.all(res_norms < 1e-4)

    def testSinglePassG(self):
        d,U = singlePassG(self.Aop,self.rhs_G,self.rhs_Ginv,self.Omega,self.k_evec,s=2)
        nvec  = U.nvec()
        AU = MultiVector(U[0], nvec)
        BU = MultiVector(U[0],nvec)
        MatMvMult(self.Aop, U, AU)
        MatMvMult(self.rhs_G,U,BU)

        # Residual checks
        diff = MultiVector(AU)
        diff.axpy(-d, BU)
        res_norms = diff.norm("l2")
        # B-orthogonality
        UtBU = BU.dot_mv(U)
        err = UtBU - np.eye(nvec, dtype=UtBU.dtype)
        err_Bortho = np.linalg.norm(err, 'fro')
        # A-orthogonality
        V = MultiVector(U)
        scaling = np.power(np.abs(d), -0.5)
        V.scale(scaling)
        AU.scale(scaling)
        VtAV = AU.dot_mv(V)
        err = VtAV - np.diag(np.sign(d))
        err_Aortho = np.linalg.norm(err, 'fro')

        if self.mpi_rank == 0:
            assert err_Bortho < 1e-8
            assert err_Aortho < 1e-4
            assert np.all(res_norms < 1e-4)

    def testDoublePassG(self):
        d,U = doublePassG(self.Aop,self.rhs_G,self.rhs_Ginv,self.Omega,self.k_evec,s=2)
        nvec  = U.nvec()
        AU = MultiVector(U[0], nvec)
        BU = MultiVector(U[0],nvec)
        MatMvMult(self.Aop, U, AU)
        MatMvMult(self.rhs_G,U,BU)

        # Residual checks
        diff = MultiVector(AU)
        diff.axpy(-d, BU)
        res_norms = diff.norm("l2")
        # B-orthogonality
        UtBU = BU.dot_mv(U)
        err = UtBU - np.eye(nvec, dtype=UtBU.dtype)
        err_Bortho = np.linalg.norm(err, 'fro')
        # A-orthogonality
        V = MultiVector(U)
        scaling = np.power(np.abs(d), -0.5)
        V.scale(scaling)
        AU.scale(scaling)
        VtAV = AU.dot_mv(V)
        err = VtAV - np.diag(np.sign(d))
        err_Aortho = np.linalg.norm(err, 'fro')

        if self.mpi_rank == 0:
            assert err_Bortho < 1e-8
            assert err_Aortho < 1e-4
            assert np.all(res_norms < 1e-4)



        

if __name__ == '__main__':
    unittest.main()
