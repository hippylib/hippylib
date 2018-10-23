# Copyright (c) 2016-2018, The University of Texas at Austin & University of
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
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import numpy as np

from .misfit import Misfit
from .timeDependentVector import TimeDependentVector
from .timeDependentOperator import TimeDependentOperator
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linalg import Transpose 
from ..utils.vector2function import vector2Function
from ..utils.checkDolfinVersion import dlversion

      
class PointwiseStateObservationTD(Misfit):
    """This class implements pointwise state observations at given locations.
    It assumes that the state variable is a scalar function.
    """
    def __init__(self, t_init, t_final, dt, misfit, t_obs=None) :
        """
        - Vh is the finite element space for the state variable
        - t_obs: final observation time (< t_final) 
        """
        self.t_init = t_init
        self.t_obs = t_obs
        self.t_final = t_final
        self.dt = dt
        self.misfit = misfit
        self.sim_times = np.arange(self.t_init, self.t_final + 0.5*self.dt, self.dt)
        self.B = TimeDependentOperator(self.misfit.B)
        self.d = TimeDependentVector(self.sim_times)
        self.B.init_vector(self.d, 0)
        self.Bu = TimeDependentVector(self.sim_times)
        self.B.init_vector(self.Bu, 0)
        self.noise_variance = misfit.noise_variance
        
    def cost(self,x):
        self.B.mult(x[STATE], self.Bu)
        self.Bu.axpy(-1., self.d)
        self.Bu.data[0].zero()
        if self.t_obs != None:
            for i, t in enumerate(self.Bu.times):
                if t > self.t_obs + 0.5*self.dt:
                    self.Bu.data[i].zero()
        
        return (.5*self.dt/self.noise_variance)*self.Bu.inner(self.Bu)
        
    def grad(self, i, x, out):
        if i == STATE:
            self.B.mult(x[STATE], self.Bu)
            self.Bu.axpy(-1., self.d)
            self.B.transpmult(self.Bu, out)
            self.Bu.data[0].zero()
            if self.t_obs != None:
                for i, t in enumerate(self.Bu.times):
                    if t > self.t_obs + 0.5*self.dt:
                        self.Bu.data[i].zero()
            out *= self.dt/self.noise_variance

        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()
                
    def setLinearizationPoint(self,x, gauss_newton_approx = False):
        return
       
    def apply_ij(self,i,j,dir,out):
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            self.B.transpmult(self.Bu, out)
            out *= self.dt/self.noise_variance
        else:
            out.zero()

class ContinuousStateObservation(Misfit):
    """This class implements continuous state observations in a
       subdomain X \subset \Omega or X \subset \partial\Omega.
    """
    def __init__(self, Vh, dX, bcs, form = None):
        """
        Constructor:
            - Vh: the finite element space for the state variable.
            - dX: the integrator on subdomain X where observation are presents.
            E.g. dX = dl.dx means observation on all \Omega and dX = dl.ds means observations on all \partial \Omega.
            - bcs: If the forward problem imposes Dirichlet boundary conditions u = u_D on \Gamma_D;
                bcs is a list of dl.DirichletBC object that prescribes homogeneuos Dirichlet conditions u = 0 on \Gamma_D.
            - form: if form = None we compute the L^2(X) misfit:
            \int_X (u - ud)^2 dX,
            otherwise the integrand specified in form will be used.
        """
        if form is None:
            u, v = dl.TrialFunction(Vh), dl.TestFunction(Vh)
            self.Wstatic = dl.assemble(dl.inner(u,v)*dX) 
            self.W = TimeDependentOperator( self.Wstatic )
        else:
            self.Wstatic = dl.assemble(form)
            self.W = TimeDependentOperator( self.Wstatic )
           
        if bcs is None:
            bcs  = []
        if isinstance(bcs, dl.DirichletBC):
            bcs = [bcs]
            
        if len(bcs):
            Wt = Transpose(self.Wstatic)
            [bc.zero(Wt) for bc in bcs]
            self.Wstatic = Transpose(Wt)
            [bc.zero(self.Wstatic) for bc in bcs]
            self.W = TimeDependentOperator( self.Wstatic)
                
        self.d = dl.Vector()
        self.W.init_vector(self.d,1)
        self.noise_variance = 0
        
    def cost(self,x):
        r = self.d.copy()
        r.axpy(-1., x[STATE])
        Wr = TimeDependentVector(r.times)
        self.W.init_vector(Wr,0)
        self.W.mult(r,Wr)
        return r.inner(Wr)/(2.*self.noise_variance)
    
    def grad(self, i, x, out):
        if i == STATE:
            self.W.mult(x[STATE]-self.d, out)
            out *= 1./self.noise_variance
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()
                   
    def apply_ij(self,i,j,dir,out):
        if i == STATE and j == STATE:
            self.W.mult(dir, out)
            out *= 1./self.noise_variance
        else:
            out.zero() 
