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
                
        true_sub = dl.CompiledSubDomain("x[0] <= .5")
        true_marker = dl.MeshFunction('size_t',mesh, 1, value=0)
        true_sub.mark(true_marker, 1)

        
        np_sub = np.ones(n, dtype=np.uint)
        np_sub[n//2:] = 0
        h = 1./n
        marker = dl.MeshFunction('size_t',mesh, 1)
        numpy2MeshFunction(mesh,[h],np_sub, marker)
                
        assert_allclose(marker.array(), true_marker.array(), rtol=1e-7, atol=1e-9 )
        
    def test2D(self):
        nx = 10
        ny = 15
        mesh = dl.UnitSquareMesh(nx, ny)

        
        true_sub = dl.CompiledSubDomain("x[0] <= .5")
        true_marker = dl.MeshFunction('size_t',mesh, 2, value=0)
        true_sub.mark(true_marker, 1)

        
        np_sub_x = np.ones(nx, dtype=np.uint)
        np_sub_x[nx//2:] = 0
        np_sub_y = np.ones(ny, dtype=np.uint)
        np_sub_xx, np_sub_yy = np.meshgrid(np_sub_x, np_sub_y, indexing='ij')
        np_sub = np_sub_xx*np_sub_yy
        h = np.array([1./nx, 1./ny])
        marker = dl.MeshFunction('size_t',mesh, 2)
        numpy2MeshFunction(mesh,h,np_sub, marker)

        assert_allclose(marker.array(), true_marker.array(), rtol=1e-7, atol=1e-9 )
        
    def test3D(self):
        nx = 10
        ny = 15
        nz = 20
        mesh = dl.UnitCubeMesh(nx, ny, nz)
                
        true_sub = dl.CompiledSubDomain("x[0] <= .5")
        true_marker = dl.MeshFunction('size_t',mesh, 3, value=0)
        true_sub.mark(true_marker, 1)

        
        np_sub_x = np.ones(nx, dtype=np.uint)
        np_sub_x[nx//2:] = 0
        np_sub_y = np.ones(ny, dtype=np.uint)
        np_sub_z = np.ones(nz, dtype=np.uint)
        
        np_sub_xx, np_sub_yy, np_sub_zz = np.meshgrid(np_sub_x, np_sub_y, np_sub_z,  indexing='ij')
        np_sub = np_sub_xx*np_sub_yy*np_sub_zz
        h = np.array([1./nx, 1./ny, 1./nz])
        marker = dl.MeshFunction('size_t',mesh, 3)
        numpy2MeshFunction(mesh,h,np_sub, marker)

        assert_allclose(marker.array(), true_marker.array(), rtol=1e-7, atol=1e-9 )

        
if __name__ == '__main__':
    unittest.main()