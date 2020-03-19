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
from hippylib import *

class TestNumpy2MeshFunction(unittest.TestCase):
        
    def test1D(self):
        n = 10
        mesh = dl.UnitIntervalMesh(n)
        Vh   = dl.FunctionSpace(mesh, "CG", 2)
        v = dl.interpolate(dl.Expression(".5*x[0]*x[0]", degree=2), Vh)
        
        true_sub = dl.CompiledSubDomain("x[0] <= .5")
        true_marker = dl.MeshFunction('size_t',mesh, 1, value=0)
        true_sub.mark(true_marker, 1)
        true_dx = dl.Measure("dx", subdomain_data=true_marker)
        true_val = dl.assemble( v*true_dx(1) )
        
        np_sub = np.ones(n, dtype=np.uint)
        np_sub[n//2:] = 0
        h = 1./n
        marker = dl.MeshFunction('size_t',mesh, 1)
        numpy2MeshFunction(mesh,[h],np_sub, marker)
        dx = dl.Measure("dx", subdomain_data=marker)
        val = dl.assemble( v*dx(1) )
        
        

        
        assert_allclose([val], [true_val], rtol=1e-7, atol=1e-9)
        
    def test2D(self):
        nx = 10
        ny = 15
        mesh = dl.UnitSquareMesh(nx, ny)

        Vh   = dl.FunctionSpace(mesh, "CG", 2)
        v = dl.interpolate(dl.Expression(".5*x[0]*x[0]+x[1]*x[1]", degree=2), Vh)
        
        true_sub = dl.CompiledSubDomain("x[0] <= .5")
        true_marker = dl.MeshFunction('size_t',mesh, 2, value=0)
        true_sub.mark(true_marker, 1)
        true_dx = dl.Measure("dx", subdomain_data=true_marker)
        true_val = dl.assemble( v*true_dx(1) )
        
        np_sub_x = np.ones(nx, dtype=np.uint)
        np_sub_x[nx//2:] = 0
        np_sub_y = np.ones(ny, dtype=np.uint)
        np_sub = np.meshgrid(np_sub_x, np_sub_y, indexing='ij')
        h = [1./nx, 1./ny]
        marker = dl.MeshFunction('size_t',mesh, 2)
        numpy2MeshFunction(mesh,h*np.ones(2),np_sub, marker)
        dx = dl.Measure("dx", subdomain_data=marker)
        val = dl.assemble( v*dx(1) )
        
        
        assert_allclose([val], [true_val], rtol=1e-7, atol=1e-9)
        
    def test3D(self):
        nx = 10
        ny = 15
        nz = 20
        mesh = dl.UnitCubeMesh(nx, ny, nz)

        val = 0.
        true_val = 0.
        assert_allclose([val], [true_val], rtol=1e-7, atol=1e-9)
        
if __name__ == '__main__':
    unittest.main()