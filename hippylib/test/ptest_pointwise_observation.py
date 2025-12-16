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

from hippylib import *

class TestPointwiseObservation(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(10,10)
        np.random.seed(1)
        self.ntargets = 10
        self.ndim     = 2
        self.targets = np.random.uniform(low = 0.1, high = 0.9,
                                         size = (self.ntargets, self.ndim))
        
        self.rank = dl.MPI.rank(self.mesh.mpi_comm())
        
    def testScalarObservations(self):
        Vh = dl.FunctionSpace(self.mesh, "CG", 1)
        xvect = dl.interpolate(dl.Expression("x[0]", degree=1), Vh).vector()
        
        B = assemblePointwiseObservation(Vh, self.targets, prune_and_sort=False)
        out = dl.Vector()
        B.init_vector(out, 0)
        B.mult(xvect, out)
        
        out_np = out.gather_on_zero()
        
        if self.rank == 0:
            assert_allclose(self.targets[:,0], out_np)
        
    def testVectorObservations(self):
        Vh = dl.VectorFunctionSpace(self.mesh, "CG", 2)
        xvect = dl.interpolate(dl.Expression(("x[0]", "x[1]"), degree=1), Vh).vector()
        
        B = assemblePointwiseObservation(Vh, self.targets, prune_and_sort=False)
        out = dl.Vector()
        B.init_vector(out,0)
        B.mult(xvect, out)
        
        out_np =  out.gather_on_zero()
        
        if self.rank == 0:
            assert_allclose(self.targets, np.reshape(out_np, (self.ntargets, self.ndim), 'C'))

    def testLOSObservations(self):
        Vh = dl.VectorFunctionSpace(self.mesh, "CG", 2)
        xvect = dl.interpolate(dl.Expression(("x[0]", "x[1]"), degree=1), Vh).vector()

        los_coeff = np.ones_like(self.targets)
        
        B = assemblePointwiseLOSObservation(Vh, self.targets, los_coeff, prune_and_sort=False)
        out = dl.Vector()
        B.init_vector(out,0)
        B.mult(xvect, out)
        
        out_np =  out.gather_on_zero()
        
        if self.rank == 0:
            assert_allclose(np.sum(self.targets*los_coeff,1), out_np)


    def testVectorComponentObservations(self):
        Vh = dl.VectorFunctionSpace(self.mesh, "CG", 2, 5)
        xvect = dl.interpolate(dl.Expression(("x[0]", "x[1]", "x[0]*x[0]", "x[1]*x[0]", "x[1]*x[1]"), degree=1), Vh).vector()
        components = np.array([0,2])
        B = assemblePointwiseObservation(Vh, self.targets, components, prune_and_sort=False)
        out = dl.Vector()
        B.init_vector(out,0)
        B.mult(xvect, out)
        
        out_np =  out.gather_on_zero()
        
        if self.rank == 0:
            out_true = np.zeros((self.targets.shape[0], components.shape[0]))
            out_true[:,0] = self.targets[:,0]
            out_true[:,1] = self.targets[:,0]*self.targets[:,0]
            assert_allclose(out_true, np.reshape(out_np, (self.ntargets, self.ndim), 'C'))
            
    def testRTObservations(self):
        Vh = dl.FunctionSpace(self.mesh, "RT", 1)
        xvect = dl.interpolate(dl.Expression(("x[0]", "x[1]"), degree=1), Vh).vector()
        
        B = assemblePointwiseObservation(Vh, self.targets, prune_and_sort=False)
        out = dl.Vector()
        B.init_vector(out,0)
        B.mult(xvect, out)
        
        out_np =  out.gather_on_zero()
        
        if self.rank == 0:
            assert_allclose(self.targets, np.reshape(out_np, (self.ntargets, self.ndim), 'C'))

    def testRTComponentObservations(self):
        Vh = dl.FunctionSpace(self.mesh, "RT", 1)
        xvect = dl.interpolate(dl.Expression(("x[0]", "x[1]"), degree=1), Vh).vector()
        components = np.array([1])
        
        B = assemblePointwiseObservation(Vh, self.targets, components, prune_and_sort=False)
        out = dl.Vector()
        B.init_vector(out,0)
        B.mult(xvect, out)
        
        out_np =  out.gather_on_zero()
        
        if self.rank == 0:
            assert_allclose(self.targets[:,1], out_np)

if __name__ == '__main__':
    unittest.main()