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
from hippylib import (MultiVector, Random, assemblePointwiseObservation, amg_method,
                      singlePass, singlePassG, doublePass, doublePassG, MatMvMult, PETScKrylovSolver)
                      


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
        varfA = ufl.inner(ufl.grad(uh), ufl.grad(vh))*ufl.dx +\
                    alpha*ufl.inner(uh,vh)*ufl.dx
        A = dl.assemble(varfA)
        Asolver = PETScKrylovSolver(A.mpi_comm(), "cg", amg_method())
        Asolver.set_operator(A)
        Asolver.parameters["maximum_iterations"] = 100
        Asolver.parameters["relative_tolerance"] = 1e-12

        ## Set up C
        varfC = ufl.inner(mh,vh)*ufl.dx
        C = dl.assemble(varfC)

        self.Hop = Hop(B, Asolver, C)

        ## Set up RHS Matrix M.
        varfM = ufl.inner(mh,test_mh)*ufl.dx
        self.M = dl.assemble(varfM)
        self.Minv = PETScKrylovSolver(self.M.mpi_comm(), "cg", amg_method())
        self.Minv.set_operator(self.M)
        self.Minv.parameters["maximum_iterations"] = 100
        self.Minv.parameters["relative_tolerance"] = 1e-12

        myRandom = Random(self.mpi_rank, self.mpi_size)

        x_vec = dl.Vector(mesh.mpi_comm())
        self.Hop.init_vector(x_vec,1)

        k_evec = 10
        p_evec = 50
        self.Omega = MultiVector(x_vec,k_evec+p_evec)
        self.k_evec = k_evec

        myRandom.normal(1.,self.Omega)

        
    def testSinglePass(self):
        d,U = singlePass(self.Hop,self.Omega,self.k_evec,s=2)
        nvec  = U.nvec()
        AU = MultiVector(U[0], nvec)
        MatMvMult(self.Hop, U, AU)

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

        assert err_Bortho < 1e-8
        assert err_Aortho < 1e-4
        assert np.all(res_norms < 1e-4)

    def testDoublePass(self):
        d,U = doublePass(self.Hop,self.Omega,self.k_evec,s=2)
        nvec  = U.nvec()
        AU = MultiVector(U[0], nvec)
        MatMvMult(self.Hop, U, AU)

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

        assert err_Bortho < 1e-8
        assert err_Aortho < 1e-4
        assert np.all(res_norms < 1e-4)

    def testSinglePassG(self):
        d,U = singlePassG(self.Hop,self.M,self.Minv,self.Omega,self.k_evec,s=2)
        nvec  = U.nvec()
        AU = MultiVector(U[0], nvec)
        BU = MultiVector(U[0],nvec)
        MatMvMult(self.Hop, U, AU)
        MatMvMult(self.M,U,BU)

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

        assert err_Bortho < 1e-8
        assert err_Aortho < 1e-4
        assert np.all(res_norms < 1e-4)

    def testDoublePassG(self):
        d,U = doublePassG(self.Hop,self.M,self.Minv,self.Omega,self.k_evec,s=2)
        nvec  = U.nvec()
        AU = MultiVector(U[0], nvec)
        BU = MultiVector(U[0],nvec)
        MatMvMult(self.Hop, U, AU)
        MatMvMult(self.M,U,BU)

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

        assert err_Bortho < 1e-8
        assert err_Aortho < 1e-4
        assert np.all(res_norms < 1e-4)
 

if __name__ == '__main__':
    unittest.main()
