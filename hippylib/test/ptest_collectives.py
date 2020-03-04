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
from hippylib import MultiVector
from hippylib import splitCommunicators
from hippylib import checkMeshConsistentPartitioning

class TestMultipleSerialPDEsCollective(unittest.TestCase):
    def setUp(self):
        self.mpi_rank = dl.MPI.rank(dl.MPI.comm_world)
        self.mpi_size = dl.MPI.size(dl.MPI.comm_world)

        self.mesh_constructor_comm = dl.MPI.comm_self

        if self.mpi_size > 1:
            self.collective = cl.MultipleSerialPDEsCollective(dl.MPI.comm_world)
        else:
            self.collective = cl.NullCollective()
        

    def testfloat(self):
        # test allReduce
        a = 1.
        a_sum = self.collective.allReduce(a,'sum')
        a_avg = self.collective.allReduce(a,'avg')

        assert_allclose( [a_sum], [float(self.mpi_size)])
        assert_allclose( [a_avg], [1.] )

        # test bcast
        if self.mpi_rank == 0:
            b = 1.
        else:
            b = 0.
        b = self.collective.bcast(b,root = 0)

        assert_allclose([b], [1.])

    def testint(self):
        # test allReduce
        a = 1
        a_sum = self.collective.allReduce(a,'sum')
        a_avg = self.collective.allReduce(a,'avg')

        assert_allclose( [a_sum], self.mpi_size)
        assert_allclose( [a_avg], [1.] )

        # test bcast
        if self.mpi_rank == 0:
            b = 1
        else:
            b = 0
        b = self.collective.bcast(b,root = 0)

        assert_allclose([b], [1.])

    def testndarray(self):
        # test allReduce 
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

        # test bcast
        if self.mpi_rank == 0:
            b = np.ones(10)
        else:
            b = np.zeros(10)
        b = self.collective.bcast(b,root = 0)

        assert_allclose(b, np.ones(10) )

    def testdlVector(self):
        # test allReduce
        mesh = dl.UnitSquareMesh(self.mesh_constructor_comm,10, 10)
        Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
        
        x_str = ["(x[0]*x[1])", "(x[1])", "(x[0])", "(x[1]+2*x[0])", "(x[0]*x[0])", "(x[1]*x[1])"]
        x_str_sum = "(x[0]*x[1])"
        
        for x_str_i in x_str[1:self.collective.size()]:
            x_str_sum = x_str_sum + "+" + x_str_i
            
        x_expression = [dl.Expression(x_str_i, degree=2) for x_str_i in x_str]
        x_expression_sum = dl.Expression(x_str_sum, degree=2)

        x_true_sum = dl.interpolate(x_expression_sum, Vh).vector()
        
        x     = dl.interpolate(x_expression[self.collective.rank()], Vh).vector()
        x_sum = self.collective.allReduce(x,'sum')
        
        diff1 = x_sum - x_true_sum
        assert_allclose( [diff1.norm("l2")], [0.], rtol=1e-7, atol=1e-12)
        # x must be overwritten
        diff2 = x - x_true_sum
        assert_allclose( [diff2.norm("l2")], [0.], rtol=1e-7, atol=1e-12)
        
        x     = dl.interpolate(x_expression[self.collective.rank()], Vh).vector()
        x_avg = self.collective.allReduce(x,'avg')
        
        x_true_avg = x_true_sum.copy()
        x_true_avg *= 1./float(self.collective.size())
        
        diff1 = x_avg - x_true_avg
        assert_allclose( [diff1.norm("l2")], [0.], rtol=1e-7, atol=1e-12)
        # x must be overwritten
        diff2 = x - x_true_avg
        assert_allclose( [diff2.norm("l2")], [0.], rtol=1e-7, atol=1e-12)

        # test bcast
        x     = dl.interpolate(x_expression[self.collective.rank()], Vh).vector()

        x = self.collective.bcast(x,root = 0)

        x_true = dl.interpolate(x_expression[0], Vh).vector()

        diff = x - x_true
        assert_allclose( [diff.norm("l2")], [0.], rtol=1e-7, atol=1e-12)

    def testMultiVector(self):
        # test allReduce
        mesh = dl.UnitSquareMesh(self.mesh_constructor_comm,10, 10)
        Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)

        x = dl.interpolate(dl.Constant(1.), Vh).vector()
        ones = np.ones_like(x.get_local())
        MV = MultiVector(x,10)
        MV_ref = MultiVector(x,10)
        for i in range(MV.nvec()):
            MV[i].set_local(ones)
            MV[i].apply("")
            MV_ref[i].set_local(ones)
            MV_ref[i].apply("")
        MV_sum = self.collective.allReduce(MV,'sum')

        # MV gets overwritten in the collective
        MV = MultiVector(x,10)
        for i in range(MV.nvec()):
            MV[i].set_local(ones)
            MV[i].apply("")
        MV_avg = self.collective.allReduce(MV,'avg')

        for i in range(MV.nvec()):
            diff1 = float(self.mpi_size)*MV_ref[i] - MV_sum[i]
            assert_allclose( [diff1.norm("l2")], [0.])
            diff2 = MV[i] - MV_avg[i]
            assert_allclose( [diff2.norm("l2")], [0.])

        # test MultiVector bcast
        MV = MultiVector(x,10)
        MV_ref = MultiVector(x,10)
        ones = np.ones_like(x.get_local())
        if self.mpi_rank == 0:
            for i in range(MV.nvec()):
                MV[i].set_local(ones)
                MV[i].apply("")
                MV_ref[i].set_local(ones)
                MV_ref[i].apply("")
        else:
            MV.zero()
            for i in range(MV_ref.nvec()):
                MV_ref[i].set_local(ones)
                MV_ref[i].apply("")

        MV = self.collective.bcast(MV,root = 0)

        for i in range(MV.nvec()):
            diff = MV[i] - MV_ref[i]
            assert_allclose( [diff.norm("l2")], [0.])
            
    def checkConsistentPartitioning(self):
        self.assertTrue(1==1)

class TestMultipleSamePartitioningPDEsCollective(TestMultipleSerialPDEsCollective):
    def setUp(self):
        self.world_rank = dl.MPI.rank(dl.MPI.comm_world)
        self.world_size = dl.MPI.size(dl.MPI.comm_world)

        if self.world_size > 1:
            assert self.world_size == 6
            n_subdomain = 2
            n_instances = 3
            self.mesh_constructor_comm, collective_comm = splitCommunicators(dl.MPI.comm_world, n_subdomain, n_instances)
            self.mpi_size = collective_comm.size
            self.mpi_rank = collective_comm.rank
            self.collective = cl.MultipleSamePartitioningPDEsCollective(collective_comm)
        else:
            self.mpi_size = dl.MPI.comm_world.size
            self.mpi_rank = dl.MPI.comm_world.rank
            self.mesh_constructor_comm = dl.MPI.comm_world
            self.collective = cl.NullCollective()

    def checkConsistentPartitioning(self):
        mesh = dl.UnitSquareMesh(self.mesh_constructor_comm,10, 10)
        consistent_partitioning = checkMeshConsistentPartitioning(mesh,self.collective)
        print('consistent paritioning: ', consistent_partitioning)
        self.assertTrue(consistent_partitioning)



if __name__ == '__main__':
    unittest.main()
