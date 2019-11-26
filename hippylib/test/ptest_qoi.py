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

class TestPointwiseObservation(unittest.TestCase):
    def setUp(self):
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
            return dl.exp(m)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx

        self.pde = PDEVariationalProblem(self.Vh, pde_varf, bc, bc0, is_fwd_linear=True)
         
        gamma = .1
        delta = .5

        self.prior = BiLaplacianPrior(self.Vh[PARAMETER], gamma, delta)
        
        GC = GammaCenter()
        marker = dl.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        marker.set_all(0)
        GC.mark(marker, 1)
        dss = dl.Measure("dS", domain=self.mesh, subdomain_data=marker)
        n = dl.Constant((0.,1.))#dl.FacetNormal(Vh[STATE].mesh())

        def qoi_varf( x):
            return dl.avg(dl.exp(x[PARAMETER])*dl.dot( dl.grad(x[STATE]), n) )*dss

        self.qoi = VariationalQOI(self.Vh,qoi_varf) 

        
    def testVariationalQOI(self):

        rqoi = ReducedQOI(self.pde, self.qoi)
        
        out = reducedQOIVerify(rqoi, self.prior.mean, eps=np.power(.5, np.arange(20,0,-1)),\
                                                                 plotting = False, verbose = False )

        assert np.all(out['err_grad'] < 1.)
        assert np.all(out['err_H']< 1.)
        assert out['rel_sym_error'] < 1e-10


if __name__ == '__main__':
    unittest.main()
