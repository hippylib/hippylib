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

class TestMultiVector(unittest.TestCase):
    def setUp(self):
        mesh = dl.UnitSquareMesh(10, 10)
        Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
        uh,vh = dl.TrialFunction(Vh),dl.TestFunction(Vh)
        varfM = dl.inner(uh,vh)*dl.dx
        M = dl.assemble(varfM)
        x = dl.Vector()
        M.init_vector(x,0)
        k = 121
        self.Q = MultiVector(x,k)

        self.rank = dl.MPI.rank(mesh.mpi_comm())
        
    def testOrthogonalization(self):
        myRandom = Random()
        myRandom.normal(1.,self.Q)
        _ = self.Q.orthogonalize()
        QtQ = self.Q.dot_mv(self.Q)

        if self.rank == 0:
            assert np.linalg.norm(QtQ - np.eye(QtQ.shape[0])) < 1e-8
        

if __name__ == '__main__':
    unittest.main()