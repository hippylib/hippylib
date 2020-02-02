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
from hippylib import MultiVector, Random, MatMvMult

class TestMultiVector(unittest.TestCase):
    def setUp(self):
        mesh = dl.UnitSquareMesh(10, 10)
        self.mpi_rank = dl.MPI.rank(mesh.mpi_comm())
        self.mpi_size = dl.MPI.size(mesh.mpi_comm())
        
        Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
        uh,vh = dl.TrialFunction(Vh),dl.TestFunction(Vh)
        
        varfM = ufl.inner(uh,vh)*ufl.dx
        self.M = dl.assemble(varfM)
        x = dl.Vector( mesh.mpi_comm() )
        self.M.init_vector(x,0)
        k = 121
        self.Q = MultiVector(x,k)
    
    def testOrthogonalization(self):
        myRandom = Random(self.mpi_rank, self.mpi_size)
        myRandom.normal(1.,self.Q)
        _ = self.Q.orthogonalize()
        QtQ = self.Q.dot_mv(self.Q)

        if self.mpi_rank == 0:
            assert np.linalg.norm(QtQ - np.eye(QtQ.shape[0])) < 1e-8
            
    def testBOrthogonalization(self):
        myRandom = Random(self.mpi_rank, self.mpi_size)
        myRandom.normal(1.,self.Q)
        self.Q.Borthogonalize(self.M)
        
        MQ = MultiVector(self.Q)
        MQ.zero()
        MatMvMult(self.M, self.Q, MQ)
        
        QtMQ = self.Q.dot_mv(MQ)

        if self.mpi_rank == 0:
            assert np.linalg.norm(QtMQ - np.eye(QtMQ.shape[0])) < 1e-8
        

if __name__ == '__main__':
    unittest.main()
