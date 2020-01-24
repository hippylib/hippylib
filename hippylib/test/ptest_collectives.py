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

from hippylib import scheduling as cl

class TestCollectives(unittest.TestCase):
    def setUp(self):
        self.collective = cl.MultipleSerialPDEsCollective(dl.MPI.comm_world)
        self.mpi_rank = dl.MPI.rank(dl.MPI.comm_world)
        self.mpi_size = dl.MPI.size(dl.MPI.comm_world)
        

    def testfloat(self):
        a = 1.
        a_sum = self.collective.allReduce(a,'sum')
        a_avg = self.collective.allReduce(a,'avg')
        if self.mpi_rank == 0:
            assert a_sum == float(self.mpi_size)
            assert a_avg == 1.

    def testint(self):
        a = 1
        a_sum = self.collective.allReduce(a,'sum')
        a_avg = self.collective.allReduce(a,'avg')
        if self.mpi_rank == 0:
            assert a_sum == self.mpi_size
            assert a_avg == 1


    def testndarray(self):
        a = np.ones(10)
        a_sum = self.collective.allReduce(a,'sum')
        a = np.ones(10)
        a_avg = self.collective.allReduce(a,'avg')
        if self.mpi_rank == 0:
            assert (a_sum == float(self.mpi_size)*np.ones(10)).all()
            assert (a_avg == np.ones(10)).all()


    def testdlVector(self):
        mesh = dl.UnitSquareMesh(dl.MPI.comm_self,10, 10)
        Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
        x = dl.interpolate(dl.Constant(1.), Vh).vector()
        x_sum = self.collective.allReduce(x,'sum')
        x = dl.interpolate(dl.Constant(1.), Vh).vector()
        diff = x_sum - self.mpi_size*x
        if self.mpi_rank == 0:
            assert np.linalg.norm(diff.get_local()) < 1e-10



if __name__ == '__main__':
    unittest.main()
