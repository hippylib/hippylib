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
from hippylib import *

class GammaCenter(dl.SubDomain):
    def inside(self, x, on_boundary):
        return ( abs(x[1]-.5) < dl.DOLFIN_EPS )

def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

class TestVariationalQoi(unittest.TestCase):
    def setUp(self):
        dl.parameters["ghost_mode"] = "shared_facet"
        ndim = 2
        nx = 10
        ny = 10
        self.mesh = dl.UnitSquareMesh(nx, ny)
        
        self.rank = dl.MPI.rank(self.mesh.mpi_comm())
            
        Vh2 = dl.FunctionSpace(self.mesh, 'Lagrange', 2)
        Vh1 = dl.FunctionSpace(self.mesh, 'Lagrange', 1)
        self.Vh = [Vh2, Vh1, Vh2]
        # Initialize Expressions
        f = dl.Constant(0.0)
            
        u_bdr = dl.Expression("x[1]", degree=1)
        u_bdr0 = dl.Constant(0.0)
        bc = dl.DirichletBC(self.Vh[STATE], u_bdr, u_boundary)
        bc0 = dl.DirichletBC(self.Vh[STATE], u_bdr0, u_boundary)
        
        def pde_varf(u,m,p):
            return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx

        self.pde = PDEVariationalProblem(self.Vh, pde_varf, bc, bc0, is_fwd_linear=True)
                 
        GC = GammaCenter()
        marker = dl.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        marker.set_all(0)
        GC.mark(marker, 1)
        dss = dl.Measure("dS", domain=self.mesh, subdomain_data=marker)
        n = dl.Constant((0.,1.))#dl.FacetNormal(Vh[STATE].mesh())

        def qoi_varf(u,m):
            return ufl.avg(ufl.exp(m)*ufl.dot( ufl.grad(u), n) )*dss(1)

        self.qoi = VariationalQoi(self.Vh,qoi_varf) 

        
    def testVariationalQOI(self):

        p2qoimap = Parameter2QoiMap(self.pde, self.qoi)
        eps = np.power(.5, np.arange(20,0,-1))
        m0 = dl.interpolate(dl.Constant(0.), self.Vh[PARAMETER]).vector()
        out = parameter2QoiMapVerify(p2qoimap, m0, eps=eps,\
                                                    plotting = False, verbose = False )
        err_g = out['err_grad']
        err_H = out['err_H']
        slope_g = []
        slope_H = []
        for i in range(1,len(eps)):
            rise_g = np.log(err_g[i]) - np.log(err_g[i-1])
            rise_H = np.log(err_H[i]) - np.log(err_H[i-1])
            run = np.log(eps[i]) - np.log(eps[i-1])
            slope_g.append(rise_g/run)
            slope_H.append(rise_H/run)

        len_slopes = len(slope_g)

        slope_error_g = np.abs(slope_g - np.ones_like(slope_g)) < 1e-1
        slope_error_H = np.abs(slope_H - np.ones_like(slope_H)) < 1e-1

        within_tolerance_g = np.sum(slope_error_g)/len_slopes
        within_tolerance_H = np.sum(slope_error_H)/len_slopes

        print('within_tolerance_g = ', within_tolerance_g)
        print('within_tolerance_H = ', within_tolerance_H)

        assert within_tolerance_g > 0.65
        assert within_tolerance_H > 0.7
        assert out['rel_sym_error'] < 1e-10


if __name__ == '__main__':
    dl.parameters["ghost_mode"] = "shared_facet"
    unittest.main()
