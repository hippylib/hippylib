# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import dolfin as dl
from pointwiseObservation import assemblePointwiseObservation
from variables import STATE

class Misfit:
    """Abstract class to model the misfit componenet of the cost functional.
    In the following x will denote the variable [u, a, p], denoting respectively
    the state u, the parameter a, and the adjoint variable p.
    
    The methods in the class misfit will usually access the state u and possibly the
    parameter a. The adjoint variables will never be accessed. 
    """
    def cost(self,x):
        """Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter a are accessed. """
        
        
    def adj_rhs(self,x,rhs):
        """Evaluate the RHS for the adjoint problem.
        Only the state u and (possibly) the parameter a are accessed. """
    
    def setLinearizationPoint(self,x):
        """Set the point for linearization."""
        
    def apply_ij(self,i,j, dir, out):
        """Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the cost in direction dir."""
        
class PointwiseStateObservation(Misfit):
    """This class implements pointwise state observations at given locations.
    It assumes that the state variable is a scalar function.
    """
    def __init__(self, Vh, obs_points):
        """
        Constructor:
        - Vh is the finite element space for the state variable
        - obs_points is a 2D array number of points by geometric dimensions that stores
          the location of the observations.
        """
        self.B = assemblePointwiseObservation(Vh, obs_points)
        self.d = dl.Vector()
        self.B.init_vector(self.d, 0)
        self.Bu = dl.Vector()
        self.B.init_vector(self.Bu, 0)
        self.noise_variance = 0.
        
    def cost(self,x):
        self.B.mult(x[STATE], self.Bu)
        self.Bu.axpy(-1., self.d)
        return (.5/self.noise_variance)*self.Bu.inner(self.Bu)
    
    def adj_rhs(self, x, out):
        self.B.mult(x[STATE], self.Bu)
        self.Bu.axpy(-1., self.d)
        self.B.transpmult(self.Bu, out)
        out *= (-1./self.noise_variance)
    
    def setLinearizationPoint(self,x):
        return
       
    def apply_ij(self,i,j,dir,out):
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            self.B.transpmult(self.Bu, out)
            out *= (1./self.noise_variance)
        else:
            out.zero()
            
class ContinuousStateObservation(Misfit):
    """This class implements continuous state observations in a
       subdomain X \subset \Omega or X \subset \partial\Omega.
    """
    def __init__(self, Vh, dX, bc, form = None):
        """
        Constructor:
        - Vh: the finite element space for the state variable.
        - dX: the integrator on subdomain X where observation are presents.
        E.g. dX = dl.dx means observation on all \Omega and dX = dl.ds means observations on all \partial \Omega.
        - bc: If the forward problem imposes Dirichlet boundary conditions u = u_D on \Gamma_D;
              bc is a dl.DirichletBC object that prescribes homogeneuos Dirichlet conditions u = 0 on \Gamma_D.
        - form: if form = None we compute the L^2(X) misfit:
          \int_X (u - ud)^2 dX,
          otherwise the integrand specified in form will be used.
        """
        if form is None:
            u, v = dl.TrialFunction(Vh), dl.TestFunction(Vh)
            self.W = dl.assemble(dl.inner(u,v)*dX)
        else:
            self.W = dl.assemble( form )
                
        if bc is not None:
            Wt = Transpose(self.W)
            self.bc.zero(Wt)
            self.W = Transpose(Wt)
            self.bc.zero(self.W)
        self.d = dl.Vector()
        self.W.init_vector(self.d,1)
        self.noise_variance = 0
        
    def cost(self,x):
        r = self.d.copy()
        r.axpy(-1., x[STATE])
        Wr = dl.Vector()
        self.W.init_vector(Wr,0)
        self.W.mult(r,Wr)
        return r.inner(Wr)/(2.*self.noise_variance)
    
    def adj_rhs(self, x, out):
        r = self.d.copy()
        r.axpy(-1., x[STATE])
        self.W.mult(r, out)
        out *= (1./self.noise_variance)
       
    def apply_ij(self,i,j,dir,out):
        if i == STATE and j == STATE:
            self.W.mult(dir, out)
            out *= (1./self.noise_variance)
        else:
            out.zero() 