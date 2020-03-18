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

class TestNumpy2Expression(unittest.TestCase):
        
    def test1D(self):
        n = 10
        mesh = dl.UnitIntervalMesh(n)
        Vh = dl.FunctionSpace(mesh, "CG", 2)
        
        n_np = 2*n
        h = 1./float(n_np)
        f_np = np.linspace(0., 1., n_np+1)
        f_exp = NumpyScalarExpression1D()
        f_exp.setData(f_np, h)
        
        f_exp2 = dl.Expression("x[0]", degree=2)
        
        fh_1 = dl.interpolate(f_exp, Vh).vector()
        fh_2 = dl.interpolate(f_exp2, Vh).vector()
                
        fh_1.axpy(-1., fh_2 )
        
        error = fh_1.norm("l2")
        
        assert_allclose([error], [0.], rtol=1e-7, atol=1e-9)
        
    def test2D(self):
        nx = 10
        ny = 15
        mesh = dl.UnitSquareMesh(nx, ny)
        Vh = dl.FunctionSpace(mesh, "CG", 2)
        
        nx_np = 2*nx
        ny_np = 2*ny
        hx = 1./float(nx_np)
        hy = 1./float(ny_np)
        x_np = np.linspace(0., 1., nx_np+1)
        y_np = np.linspace(0., 1., ny_np+1)
        xx, yy = np.meshgrid(x_np, y_np, indexing='ij')
        f_np = xx + 2.*yy
        f_exp = NumpyScalarExpression2D()
        f_exp.setData(f_np, hx, hy)
        
        f_exp2 = dl.Expression("x[0] + 2.*x[1]", degree=2)
        
        fh_1 = dl.interpolate(f_exp, Vh).vector()
        fh_2 = dl.interpolate(f_exp2, Vh).vector()
                
        fh_1.axpy(-1., fh_2 )
        
        error = fh_1.norm("l2")
        
        assert_allclose([error], [0.], rtol=1e-7, atol=1e-9)
        
    def test3D(self):
        nx = 10
        ny = 15
        nz = 20
        mesh = dl.UnitCubeMesh(nx, ny, nz)
        Vh = dl.FunctionSpace(mesh, "CG", 2)
        
        nx_np = 2*nx
        ny_np = 2*ny
        nz_np = 2*nz
        hx = 1./float(nx_np)
        hy = 1./float(ny_np)
        hz = 1./float(nz_np)
        
        x_np = np.linspace(0., 1., nx_np+1)
        y_np = np.linspace(0., 1., ny_np+1)
        z_np = np.linspace(0., 1., nz_np+1)
        
        xx, yy, zz = np.meshgrid(x_np, y_np, z_np, indexing='ij')
        f_np = xx + 2.*yy - 3.*zz
        f_exp = NumpyScalarExpression3D()
        f_exp.setData(f_np, hx, hy, hz)
        
        f_exp2 = dl.Expression("x[0] + 2.*x[1] - 3.*x[2]", degree=2)
        
        fh_1 = dl.interpolate(f_exp, Vh).vector()
        fh_2 = dl.interpolate(f_exp2, Vh).vector()
                
        fh_1.axpy(-1., fh_2 )
        
        error = fh_1.norm("l2")
        
        assert_allclose([error], [0.], rtol=1e-7, atol=1e-9)
        
if __name__ == '__main__':
    unittest.main()