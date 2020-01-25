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
import numpy as np

from numpy.testing import assert_allclose

import sys
sys.path.append('../../')

from hippylib import scheduling as cl

class TestCollectives(unittest.TestCase):
    def setUp(self):
        self.mpi_rank = dl.MPI.rank(dl.MPI.comm_world)
        self.mpi_size = dl.MPI.size(dl.MPI.comm_world)

        if self.mpi_size > 1:
            self.collective = cl.MultipleSerialPDEsCollective(dl.MPI.comm_world)
        else:
            self.collective = cl.NullCollective()
        

    def testfloat(self):
        a = 1.
        a_sum = self.collective.allReduce(a,'sum')
        a_avg = self.collective.allReduce(a,'avg')

        assert_allclose( [a_sum], [float(self.mpi_size)])
        assert_allclose( [a_avg], [1.] )

    def testint(self):
        a = 1
        a_sum = self.collective.allReduce(a,'sum')
        a_avg = self.collective.allReduce(a,'avg')

        assert_allclose( [a_sum], self.mpi_size)
        assert_allclose( [a_avg], [1.] )


    def testndarray(self):
        a = np.ones(10)
        a_sum = self.collective.allReduce(a,'sum')
        
        assert_allclose(a_sum, self.mpi_size*np.ones(10) )
        # `a` must be overwritten
        assert_allclose(a   , self.mpi_size*np.ones(10) )
        
        a = np.ones(10)
        a_avg = self.collective.allReduce(a,'avg')
        
        assert_allclose(a_avg, np.ones(10) )
        # `a` must be overwritten
        assert_allclose(a   , np.ones(10) )


    def testdlVector(self):
        mesh = dl.UnitSquareMesh(dl.MPI.comm_self,10, 10)
        Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)

        x_ref = dl.interpolate(dl.Constant(1.), Vh).vector()
        
        x     = dl.interpolate(dl.Constant(1.), Vh).vector()
        x_sum = self.collective.allReduce(x,'sum')
        
        diff1 = x_sum - float(self.mpi_size)*x_ref
        assert_allclose( [diff1.norm("l2")], [0.])
        # x must be overwritten
        diff2 = x - float(self.mpi_size)*x_ref
        assert_allclose( [diff2.norm("l2")], [0.])
        
        x     = dl.interpolate(dl.Constant(1.), Vh).vector()
        x_avg = self.collective.allReduce(x,'avg')
        
        diff1 = x_avg - x_ref
        assert_allclose( [diff1.norm("l2")], [0.])
        # x must be overwritten
        diff2 = x - x_ref
        assert_allclose( [diff2.norm("l2")], [0.])
        


if __name__ == '__main__':
    unittest.main()
