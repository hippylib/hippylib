# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
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

from __future__ import absolute_import, division, print_function

import dolfin as dl
from .pointwiseObservation import assemblePointwiseObservation
from .variables import STATE, PARAMETER
from ..algorithms.linalg import Transpose

class Misfit(object):
    """
    Abstract class to model the misfit componenet of the cost functional.
    In the following :code:`x` will denote the variable :code:`[u, m, p]`, denoting respectively 
    the state :code:`u`, the parameter :code:`m`, and the adjoint variable :code:`p`.
    
    The methods in the class misfit will usually access the state u and possibly the
    parameter :code:`m`. The adjoint variables will never be accessed. 
    """
    def cost(self,x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed. """
        raise NotImplementedError("Child class should implement method cost")
        return 0
        
    def grad(self, i, x, out):
        """
        Given the state and the paramter in :code:`x`, compute the partial gradient of the misfit
        functional in with respect to the state (:code:`i == STATE`) or with respect to the parameter (:code:`i == PARAMETER`).
        """
        raise NotImplementedError("Child class should implement method grad")
            
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        """
        Set the point for linearization.

        Inputs:
        
            :code:`x=[u, m, p]` - linearization point

            :code:`gauss_newton_approx (bool)` - whether to use Gauss Newton approximation 
        """
        raise NotImplementedError("Child class should implement method setLinearizationPoint")
        
    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """
        raise NotImplementedError("Child class should implement method apply_ij")
        
class PointwiseStateObservation(Misfit):
    """
    This class implements pointwise state observations at given locations.
    It assumes that the state variable is a scalar function.
    """
    def __init__(self, Vh, obs_points):
        """
        Constructor:

            :code:`Vh` is the finite element space for the state variable
            
            :code:`obs_points` is a 2D array number of points by geometric dimensions that stores \
            the location of the observations.
        """
        self.B = assemblePointwiseObservation(Vh, obs_points)
        self.d = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.d, 0)
        self.Bu = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.Bu, 0)
        self.noise_variance = None
        
    def cost(self,x):
        if self.noise_variance is None: 
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        self.B.mult(x[STATE], self.Bu)
        self.Bu.axpy(-1., self.d)
        return (.5/self.noise_variance)*self.Bu.inner(self.Bu)
    
    def grad(self, i, x, out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE:
            self.B.mult(x[STATE], self.Bu)
            self.Bu.axpy(-1., self.d)
            self.B.transpmult(self.Bu, out)
            out *= (1./self.noise_variance)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()
                
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        # The cost functional is already quadratic. Nothing to be done here
        return
       
    def apply_ij(self,i,j,dir,out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            self.B.transpmult(self.Bu, out)
            out *= (1./self.noise_variance)
        else:
            out.zero()
            
class ContinuousStateObservation(Misfit):
    """
    This class implements continuous state observations in a 
    subdomain :math:`X \subset \Omega` or :math:`X \subset \partial \Omega`.
    """
    def __init__(self, Vh, dX, bcs, form = None):
        """
        Constructor:

            :code:`Vh`: the finite element space for the state variable.
            
            :code:`dX`: the integrator on subdomain `X` where observation are presents. \
            E.g. :code:`dX = dl.dx` means observation on all :math:`\Omega` and :code:`dX = dl.ds` means observations on all :math:`\partial \Omega`.
            
            :code:`bcs`: If the forward problem imposes Dirichlet boundary conditions :math:`u = u_D \mbox{ on } \Gamma_D`;  \
            :code:`bcs` is a list of :code:`dolfin.DirichletBC` object that prescribes homogeneuos Dirichlet conditions :math:`u = 0 \mbox{ on } \Gamma_D`.
            
            :code:`form`: if :code:`form = None` we compute the :math:`L^2(X)` misfit: :math:`\int_X (u - u_d)^2 dX,` \
            otherwise the integrand specified in the given form will be used.
        """
        if form is None:
            u, v = dl.TrialFunction(Vh), dl.TestFunction(Vh)
            self.W = dl.assemble(dl.inner(u,v)*dX)
        else:
            self.W = dl.assemble( form )
           
        if bcs is None:
            bcs  = []
        if isinstance(bcs, dl.DirichletBC):
            bcs = [bcs]
            
        if len(bcs):
            Wt = Transpose(self.W)
            [bc.zero(Wt) for bc in bcs]
            self.W = Transpose(Wt)
            [bc.zero(self.W) for bc in bcs]
                
        self.d = dl.Vector(self.W.mpi_comm())
        self.W.init_vector(self.d,1)
        self.noise_variance = None
        
    def cost(self,x):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        r = self.d.copy()
        r.axpy(-1., x[STATE])
        Wr = dl.Vector(self.W.mpi_comm())
        self.W.init_vector(Wr,0)
        self.W.mult(r,Wr)
        return r.inner(Wr)/(2.*self.noise_variance)
    
    def grad(self, i, x, out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE:
            self.W.mult(x[STATE]-self.d, out)
            out *= (1./self.noise_variance)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()
        
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        # The cost functional is already quadratic. Nothing to be done here
        return
                   
    def apply_ij(self,i,j,dir,out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE and j == STATE:
            self.W.mult(dir, out)
            out *= (1./self.noise_variance)
        else:
            out.zero() 


class MultiStateMisfit(Misfit):
    def __init__(self, misfits):
        self.nmisfits = len(misfits)
        self.misfits = misfits
        
    def append(self, misfit):
        self.nmisfits += 1
        self.misfits.append(misfit)

    def cost(self,x):
        u, m, p = x
        c = 0.
        for i in range(self.nmisfits):
            c += self.misfits[i].cost([ u.data[i], m, None ] )
        return c
    
    def grad(self, i, x, out):
        out.zero()
        u, m, p = x
        if i == STATE:
            for ii in range(self.nmisfits):
                self.misfits[ii].grad(i, [ u.data[ii], m, None ], out.data[ii] )
        else:
            tmp = out.copy()
            for ii in range(self.nmisfits):
                self.misfits[ii].grad(i, [ u.data[ii], m, None ], tmp )
                out.axpy(1., tmp)
        
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        u, m, p = x
        for ii in range(self.nmisfits):
            self.misfits[ii].setLinearizationPoint([ u.data[ii], m, None ], gauss_newton_approx)
        
       
    def apply_ij(self,i,j,dir,out):
        out.zero()
        if i == STATE and j == STATE:
            for s in range(self.nmisfits):
                self.misfits[s].apply_ij(i,j,dir.data[s],out.data[s])
        elif i == STATE and j == PARAMETER:
            for s in range(self.nmisfits):
                self.misfits[s].apply_ij(i,j,dir,out.data[s])
        elif i == PARAMETER and j == STATE:
            tmp = out.copy()
            for s in range(self.nmisfits):
                self.misfits[s].apply_ij(i,j,dir.data[s],tmp)
                out.axpy(1., tmp)
        elif i == PARAMETER and j == PARAMETER:
            tmp = out.copy()
            for s in range(self.nmisfits):
                self.misfits[s].apply_ij(i,j,dir,tmp)
                out.axpy(1., tmp)
        else:
            raise IndexError
