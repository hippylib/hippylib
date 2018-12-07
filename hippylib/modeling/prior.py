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
import numpy as np
import scipy.linalg as scila
import math

from ..algorithms.linalg import MatMatMult, get_diagonal, amg_method, estimate_diagonal_inv2, Solver2Operator, Operator2Solver
from ..algorithms.traceEstimator import TraceEstimator
from ..algorithms.multivector import MultiVector
from ..algorithms.randomizedEigensolver import singlePass, doublePass, singlePassG, doublePassG

from ..utils.checkDolfinVersion import dlversion
from ..utils.random import parRandom
from ..utils.deprecate import deprecated

from .expression import code_Mollifier

class _RinvM:
    """
    Operator that models the action of :math:`R^{-1}M`.
    It is used in the randomized trace estimator.
    """
    def __init__(self, Rsolver, M):
        self.Rsolver = Rsolver
        self.M = M
        
    def init_vector(self,x,dim):
        self.M.init_vector(x,dim)
            
    def mult(self,x,y):
        self.Rsolver.solve(y, self.M*x)

class _Prior:
    """
    Abstract class to describe the prior model.
    Concrete instances of a :code:`_Prior class` should expose
    the following attributes and methods.
    
    Attributes:

    - :code:`R`:       an operator to apply the regularization/precision operator.
    - :code:`Rsolver`: an operator to apply the inverse of the regularization/precision operator.
    - :code:`M`:       the mass matrix in the control space.
    - :code:`mean`:    the prior mean.
    
    Methods:

    - :code:`init_vector(self,x,dim)`: Inizialize a vector :code:`x` to be compatible with the range/domain of :code:`R`
      If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
      white noise used for sampling.
      
    - :code:`sample(self, noise, s, add_mean=True)`: Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample s from the prior.
      If :code:`add_mean==True` add the prior mean value to :code:`s`.
    """ 
               
    def trace(self, method="Exact", tol=1e-1, min_iter=20, max_iter=100, r = 200):
        """
        Compute/estimate the trace of the prior covariance operator.
        
        - If :code:`method=="Exact"` we compute the trace exactly by summing the diagonal entries of :math:`R^{-1}M`.
          This requires to solve :math:`n` linear system in :math:`R` (not scalable, but ok for illustration purposes).
          
        - If :code:`method=="Estimator"` use the trace estimator algorithms implemeted in the class :code:`TraceEstimator`.
          :code:`tol` is a relative bound on the estimator standard deviation. In particular, we used enough samples in the
          Estimator such that the standard deviation of the estimator is less then :code:`tol`:math:`tr(\\mbox{Prior})`.
          :code:`min_iter` and :code:`max_iter` are the lower and upper bound on the number of samples to be used for the
          estimation of the trace. 
        """
        op = _RinvM(self.Rsolver, self.M)
        if method == "Exact":
            marginal_variance = dl.Vector(self.R.mpi_comm())
            self.init_vector(marginal_variance,0)
            get_diagonal(op, marginal_variance)
            return marginal_variance.sum()
        elif method == "Estimator":
            tr_estimator = TraceEstimator(op, False, tol)
            tr_exp, tr_var = tr_estimator(min_iter, max_iter)
            return tr_exp
        elif method == "Randomized":
            dummy = dl.Vector(self.R.mpi_comm())
            self.init_vector(dummy,0)
            Omega = MultiVector(dummy, r)
            parRandom.normal(1., Omega)
            d, _ = doublePassG(Solver2Operator(self.Rsolver),
                               Solver2Operator(self.Msolver),
                               Operator2Solver(self.M),
                               Omega, r, s = 1, check = False )
            return d.sum()
        else:
            raise NameError("Unknown method")
        
    def pointwise_variance(self, method, k = 1000000, r = 200):
        """
        Compute/estimate the prior pointwise variance.
        
        - If :code:`method=="Exact"` we compute the diagonal entries of :math:`R^{-1}` entry by entry. 
          This requires to solve :math:`n` linear system in :math:`R` (not scalable, but ok for illustration purposes).
        """
        pw_var = dl.Vector(self.R.mpi_comm())
        self.init_vector(pw_var,0)
        if method == "Exact":
            get_diagonal(Solver2Operator(self.Rsolver, init_vector=self.init_vector), pw_var)
        elif method == "Estimator":
            estimate_diagonal_inv2(self.Rsolver, k, pw_var)
        elif method == "Randomized":
            Omega = MultiVector(pw_var, r)
            parRandom.normal(1., Omega)
            d, U = doublePass(Solver2Operator(self.Rsolver),
                               Omega, r, s = 1, check = False )
            
            for i in np.arange(U.nvec()):
                pw_var.axpy(d[i], U[i]*U[i])
        else:
            raise NameError("Unknown method")
        
        return pw_var
    
    def cost(self,m):
        d = self.mean.copy()
        d.axpy(-1., m)
        Rd = dl.Vector(self.R.mpi_comm())
        self.init_vector(Rd,0)
        self.R.mult(d,Rd)
        return .5*Rd.inner(d)
    
    def grad(self,m, out):
        d = m.copy()
        d.axpy(-1., self.mean)
        self.R.mult(d,out)

    def init_vector(self,x,dim):
        raise NotImplementedError("Child class should implement method init_vector")

    def sample(self, noise, s, add_mean=True):
        raise NotImplementedError("Child class should implement method sample")
        
class LaplacianPrior(_Prior):
    """
    This class implements a prior model with covariance matrix
    :math:`C = (\\delta I - \\gamma \\Delta) ^ {-1}`.
    
    The magnitude of :math:`\\gamma` governs the variance of the samples, while
    the ratio :math:`\\frac{\\gamma}{\\delta}` governs the correlation length.
    
        .. note:: :math:`C` is a trace class operator only in 1D while it is not a valid prior in 2D and 3D.
    """
    
    def __init__(self, Vh, gamma, delta, mean=None, rel_tol=1e-12, max_iter=100):
        """
        Construct the prior model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:`gamma` and :code:`delta`: the coefficient in the PDE
        - :code:`Theta`:           the SPD tensor for anisotropic diffusion of the PDE
        - :code:`mean`:            the prior mean
        """        
        assert delta != 0., "Intrinsic Gaussian Prior are not supported"
        self.Vh = Vh
        
        trial = dl.TrialFunction(Vh)
        test  = dl.TestFunction(Vh)
        
        varfL = dl.inner(dl.nabla_grad(trial), dl.nabla_grad(test))*dl.dx
        varfM = dl.inner(trial,test)*dl.dx
        
        self.M = dl.assemble(varfM)
        self.R = dl.assemble(gamma*varfL + delta*varfM)
        
        if dlversion() <= (1,6,0):
            self.Rsolver = dl.PETScKrylovSolver("cg", amg_method())
        else:
            self.Rsolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", amg_method())
        self.Rsolver.set_operator(self.R)
        self.Rsolver.parameters["maximum_iterations"] = max_iter
        self.Rsolver.parameters["relative_tolerance"] = rel_tol
        self.Rsolver.parameters["error_on_nonconvergence"] = True
        self.Rsolver.parameters["nonzero_initial_guess"] = False
        
        if dlversion() <= (1,6,0):
            self.Msolver = dl.PETScKrylovSolver("cg", "jacobi")
        else:
            self.Msolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False
        
        ndim = Vh.mesh().geometry().dim()
        old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
        dl.parameters["form_compiler"]["quadrature_degree"] = -1
        qdegree = 2*Vh._ufl_element.degree()
        metadata = {"quadrature_degree" : qdegree}
        
        if dlversion() >= (2017,1,0):
            representation_old = dl.parameters["form_compiler"]["representation"]
            dl.parameters["form_compiler"]["representation"] = "quadrature"
            
        if dlversion() <= (1,6,0):
            Qh = dl.VectorFunctionSpace(Vh.mesh(), 'Quadrature', qdegree, dim=(ndim+1) )
        else:
            element = dl.VectorElement("Quadrature", Vh.mesh().ufl_cell(),
                                       qdegree, dim=(ndim+1), quad_scheme="default")
            Qh = dl.FunctionSpace(Vh.mesh(), element)
            
        ph = dl.TrialFunction(Qh)
        qh = dl.TestFunction(Qh)
        
        pph = dl.split(ph)
        
        Mqh = dl.assemble(dl.inner(ph, qh)*dl.dx(metadata = metadata))
        ones = dl.Vector(self.R.mpi_comm())
        Mqh.init_vector(ones,0)
        ones.set_local( np.ones(ones.get_local().shape, dtype =ones.get_local().dtype ) )
        dMqh = Mqh*ones
        dMqh.set_local( ones.get_local() / np.sqrt(dMqh.get_local() ) )
        Mqh.zero()
        Mqh.set_diagonal(dMqh)
        
        sqrtdelta = math.sqrt(delta)
        sqrtgamma = math.sqrt(gamma)
        varfGG = sqrtdelta*pph[0]*test*dl.dx(metadata = metadata)
        for i in range(ndim):
            varfGG = varfGG + sqrtgamma*pph[i+1]*test.dx(i)*dl.dx(metadata = metadata)
            
        GG = dl.assemble(varfGG)
        self.sqrtR = MatMatMult(GG, Mqh)
        
        dl.parameters["form_compiler"]["quadrature_degree"] = old_qr
        
        if dlversion() >= (2017,1,0):
            dl.parameters["form_compiler"]["representation"] = representation_old
                        
        self.mean = mean
        
        if self.mean is None:
            self.mean = dl.Vector(self.R.mpi_comm())
            self.init_vector(self.mean, 0)
        
    def init_vector(self,x,dim):
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            self.sqrtR.init_vector(x,1)
        else:
            self.R.init_vector(x,dim)
                
    def sample(self, noise, s, add_mean=True):
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """

        rhs = self.sqrtR*noise
        self.Rsolver.solve(s,rhs)
        
        if add_mean:
            s.axpy(1., self.mean)
        

class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, A, Msolver):
        self.A = A
        self.Msolver = Msolver

        self.help1, self.help2 = dl.Vector(self.A.mpi_comm()), dl.Vector(self.A.mpi_comm())
        self.A.init_vector(self.help1, 0)
        self.A.init_vector(self.help2, 1)
        
    def init_vector(self,x, dim):
        self.A.init_vector(x,1)
        
    def mpi_comm(self):
        return self.A.mpi_comm()

    @deprecated(name="_BilaplacianR.inner(x, y)",
                version="2.2.0",
                msg="It will be removed in hIPPYlib 3.x\n Use _BilaplacianR.mult(x, Rx); Rx.inner(y) instead")
    def inner(self,x,y):
        Rx = dl.Vector(self.A.mpi_comm())
        self.init_vector(Rx,0)
        self.mult(x, Rx)
        return Rx.inner(y)
        
    def mult(self,x,y):
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help2, self.help1)
        self.A.mult(self.help2, y)
        
class _BilaplacianRsolver():
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, Asolver, M):
        self.Asolver = Asolver
        self.M = M
        
        self.help1, self.help2 = dl.Vector(self.M.mpi_comm()), dl.Vector(self.M.mpi_comm())
        self.init_vector(self.help1, 0)
        self.init_vector(self.help2, 0)
        
    def init_vector(self,x, dim):
        self.M.init_vector(x,1)
        
    def solve(self,x,b):
        nit = self.Asolver.solve(self.help1, b)
        self.M.mult(self.help1, self.help2)
        nit += self.Asolver.solve(x, self.help2)
        return nit
        
        
class BiLaplacianPrior(_Prior):
    """
    This class implement a prior model with covariance matrix
    :math:`C = (\\delta I + \\gamma \\mbox{div } \\Theta \\nabla) ^ {-2}`.
    
    The magnitude of :math:`\\delta\\gamma` governs the variance of the samples, while
    the ratio :math:`\\frac{\\gamma}{\\delta}` governs the correlation lenght.
    
    Here :math:`\\Theta` is a SPD tensor that models anisotropy in the covariance kernel.
    """
    
    def __init__(self, Vh, gamma, delta, Theta = None, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False):
        """
        Construct the prior model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:`gamma` and :code:`delta`: the coefficient in the PDE
        - :code:`Theta`:           the SPD tensor for anisotropic diffusion of the PDE
        - :code:`mean`:            the prior mean
        """
        assert delta != 0., "Intrinsic Gaussian Prior are not supported"
        self.Vh = Vh
        
        trial = dl.TrialFunction(Vh)
        test  = dl.TestFunction(Vh)
        
        if Theta == None:
            varfL = dl.inner(dl.nabla_grad(trial), dl.nabla_grad(test))*dl.dx
        else:
            varfL = dl.inner( Theta*dl.grad(trial), dl.grad(test))*dl.dx
        
        varfM = dl.inner(trial,test)*dl.dx
        
        varf_robin = trial*test*dl.ds
        
        if robin_bc:
            robin_coeff = gamma*np.sqrt(delta/gamma)/1.42
        else:
            robin_coeff = 0.
        
        self.M = dl.assemble(varfM)
        if dlversion() <= (1,6,0):
            self.Msolver = dl.PETScKrylovSolver("cg", "jacobi")
        else:
            self.Msolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False
        
        self.A = dl.assemble(gamma*varfL + delta*varfM + robin_coeff*varf_robin)        
        if dlversion() <= (1,6,0):
            self.Asolver = dl.PETScKrylovSolver("cg", amg_method())
        else:
            self.Asolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", amg_method())
        self.Asolver.set_operator(self.A)
        self.Asolver.parameters["maximum_iterations"] = max_iter
        self.Asolver.parameters["relative_tolerance"] = rel_tol
        self.Asolver.parameters["error_on_nonconvergence"] = True
        self.Asolver.parameters["nonzero_initial_guess"] = False
        
        old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
        dl.parameters["form_compiler"]["quadrature_degree"] = -1
        qdegree = 2*Vh._ufl_element.degree()
        metadata = {"quadrature_degree" : qdegree}

        if dlversion() >= (2017,1,0):
            representation_old = dl.parameters["form_compiler"]["representation"]
            dl.parameters["form_compiler"]["representation"] = "quadrature"
            
        if dlversion() <= (1,6,0):
            Qh = dl.FunctionSpace(Vh.mesh(), 'Quadrature', qdegree)
        else:
            element = dl.FiniteElement("Quadrature", Vh.mesh().ufl_cell(), qdegree, quad_scheme="default")
            Qh = dl.FunctionSpace(Vh.mesh(), element)
            
        ph = dl.TrialFunction(Qh)
        qh = dl.TestFunction(Qh)
        Mqh = dl.assemble(ph*qh*dl.dx(metadata=metadata))
        ones = dl.interpolate(dl.Constant(1.), Qh).vector()
        dMqh = Mqh*ones
        Mqh.zero()
        dMqh.set_local( ones.get_local() / np.sqrt(dMqh.get_local() ) )
        Mqh.set_diagonal(dMqh)
        MixedM = dl.assemble(ph*test*dl.dx(metadata=metadata))
        self.sqrtM = MatMatMult(MixedM, Mqh)

        dl.parameters["form_compiler"]["quadrature_degree"] = old_qr
        
        if dlversion() >= (2017,1,0):
            dl.parameters["form_compiler"]["representation"] = representation_old
                             
        self.R = _BilaplacianR(self.A, self.Msolver)      
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
         
        self.mean = mean
        
        if self.mean is None:
            self.mean = dl.Vector(self.R.mpi_comm())
            self.init_vector(self.mean, 0)
     
    def init_vector(self,x,dim):
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            self.sqrtM.init_vector(x, 1)
        else:
            self.A.init_vector(x,dim)   
        
    def sample(self, noise, s, add_mean=True):
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
        rhs = self.sqrtM*noise
        self.Asolver.solve(s, rhs)
        
        if add_mean:
            s.axpy(1., self.mean)
        

class MollifiedBiLaplacianPrior(_Prior):
    """
    This class implement a prior model with covariance matrix
    :math:`C = \\left( [\\delta + \\mbox{pen} \\sum_i m(x - x_i) ] I + \\gamma \\mbox{div } \\Theta \\nabla\\right) ^ {-2}`,
    
    where

    - :math:`\\Theta` is a SPD tensor that models anisotropy in the covariance kernel.
    - :math:`x_i (i=1,...,n)` are points were we assume to know exactly the value of the parameter (i.e., :math:`m(x_i) = m_{\\mbox{true}}( x_i) \\mbox{ for } i=1,...,n).`    
    - :math:`m` is the mollifier function: :math:`m(x - x_i) = \\exp\\left( - \\left[\\frac{\\gamma}{\\delta}\\| x - x_i \\|_{\\Theta^{-1}}\\right]^{\\mbox{order}} \\right).`
    - :code:`pen` is a penalization parameter.
    
    The magnitude of :math:`\\delta \\gamma` governs the variance of the samples, while
    the ratio :math:`\\frac{\\gamma}{\\delta}` governs the correlation length.
    
    The prior mean is computed by solving 
    
        .. math:: \\left( [\\delta + \\sum_i m(x - x_i) ] I + \\gamma \\mbox{div } \\Theta \\nabla \\right) m = \\sum_i m(x - x_i) m_{\\mbox{true}}.
    
    """
    
    def __init__(self, Vh, gamma, delta, locations, m_true, Theta = None, pen = 1e1, order=2, rel_tol=1e-12, max_iter=1000):
        """
        Construct the prior model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:`gamma` and :code:`delta`: the coefficients in the PDE
        - :code:`locations`:       the points :math:`x_i` at which we assume to know the true value of the parameter
        - :code:`m_true`:          the true model
        - :code:`Theta`:           the SPD tensor for anisotropic diffusion of the PDE
        - :code:`pen`:             a penalization parameter for the mollifier

        """
        assert delta != 0. or pen != 0, "Intrinsic Gaussian Prior are not supported"
        self.Vh = Vh
        
        trial = dl.TrialFunction(Vh)
        test  = dl.TestFunction(Vh)
        
        if Theta == None:
            varfL = dl.inner(dl.nabla_grad(trial), dl.nabla_grad(test))*dl.dx
        else:
            varfL = dl.inner(Theta*dl.grad(trial), dl.grad(test))*dl.dx
        varfM = dl.inner(trial,test)*dl.dx
        
        self.M = dl.assemble(varfM)
        if dlversion() <= (1,6,0):
            self.Msolver = dl.PETScKrylovSolver("cg", "jacobi")
        else:
            self.Msolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False
        
        #mfun = Mollifier(gamma/delta, dl.inv(Theta), order, locations)
        mfun = dl.Expression(code_Mollifier, degree = Vh.ufl_element().degree()+2)
        mfun.l = gamma/delta
        mfun.o = order
        mfun.theta0 = 1./Theta.theta0
        mfun.theta1 = 1./Theta.theta1
        mfun.alpha = Theta.alpha
        for ii in range(locations.shape[0]):
            mfun.addLocation(locations[ii,0], locations[ii,1])
            
        varfmo = mfun*dl.inner(trial,test)*dl.dx
        MO = dl.assemble(pen*varfmo)
                
        self.A = dl.assemble(gamma*varfL+delta*varfM + pen*varfmo)
     
        if dlversion() <= (1,6,0):
            self.Asolver = dl.PETScKrylovSolver("cg", amg_method())
        else:
            self.Asolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", amg_method())
        self.Asolver.set_operator(self.A)
        self.Asolver.parameters["maximum_iterations"] = max_iter
        self.Asolver.parameters["relative_tolerance"] = rel_tol
        self.Asolver.parameters["error_on_nonconvergence"] = True
        self.Asolver.parameters["nonzero_initial_guess"] = False
        
        old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
        dl.parameters["form_compiler"]["quadrature_degree"] = -1
        qdegree = 2*Vh._ufl_element.degree()
        metadata = {"quadrature_degree" : qdegree}
        
        if dlversion() >= (2017,1,0):
            representation_old = dl.parameters["form_compiler"]["representation"]
            dl.parameters["form_compiler"]["representation"] = "quadrature"
        
        if dlversion() <= (1,6,0):
            Qh = dl.FunctionSpace(Vh.mesh(), 'Quadrature', qdegree)
        else:
            element = dl.FiniteElement("Quadrature", Vh.mesh().ufl_cell(), qdegree, quad_scheme="default")
            Qh = dl.FunctionSpace(Vh.mesh(), element)
            
        ph = dl.TrialFunction(Qh)
        qh = dl.TestFunction(Qh)
        Mqh = dl.assemble(ph*qh*dl.dx(metadata=metadata))
        ones = dl.interpolate(dl.Constant(1.), Qh).vector()
        dMqh = Mqh*ones
        Mqh.zero()
        dMqh.set_local( ones.get_local() / np.sqrt(dMqh.get_local() ) )
        Mqh.set_diagonal(dMqh)
        MixedM = dl.assemble(ph*test*dl.dx(metadata=metadata))
        self.sqrtM = MatMatMult(MixedM, Mqh)
        
        dl.parameters["form_compiler"]["quadrature_degree"] = old_qr
        
        if dlversion() >= (2017,1,0):
            dl.parameters["form_compiler"]["representation"] = representation_old
             
        self.R = _BilaplacianR(self.A, self.Msolver)      
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
        
        rhs = dl.Vector(self.R.mpi_comm())
        self.mean = dl.Vector(self.R.mpi_comm())
        self.init_vector(rhs, 0)
        self.init_vector(self.mean, 0)
        
        MO.mult(m_true, rhs)
        self.Asolver.solve(self.mean, rhs)
        
     
    def init_vector(self,x,dim):
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            self.sqrtM.init_vector(x, 1)
        else:
            self.A.init_vector(x,dim)   
        
    def sample(self, noise, s, add_mean=True):
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
        rhs = self.sqrtM*noise
        self.Asolver.solve(s, rhs)
        
        if add_mean:
            s.axpy(1., self.mean)


class GaussianRealPrior(_Prior):
    """
    This class implements a finite-dimensional Gaussian prior,
    :math:`\\mathcal{N}(\\boldsymbol{m}, \\boldsymbol{C})`, where
    :math:`\\boldsymbol{m}` is the mean of the Gaussian distribution, and
    :math:`\\boldsymbol{C}` is its covariance. The underlying finite element
    space is assumed to be the "R" space.
    """

    def __init__(self, Vh, covariance, mean=None):
        """
        Constructor

        Inputs:
        - :code:`Vh`:             Finite element space on which the prior is
                                  defined. Must be the Real space with one global 
                                  degree of freedom
        - :code:`covariance`:     The covariance of the prior. Must be a
                                  :code:`numpy.ndarray` of appropriate size
        - :code:`mean`(optional): Mean of the prior distribution. Must be of
                                  type `dolfin.Vector()`
        """

        self.Vh = Vh

        if Vh.dim() != covariance.shape[0] or Vh.dim() != covariance.shape[1]:
            raise ValueError("Covariance incompatible with Finite Element space")

        if not np.issubdtype(covariance.dtype, np.floating):
            raise TypeError("Covariance matrix must be a float array")

        self.covariance = covariance
        
        #np.linalg.cholesky automatically provides more error checking, 
        #so use those
        self.chol = np.linalg.cholesky(self.covariance)

        self.chol_inv = scila.solve_triangular(
                                        self.chol,
                                        np.identity(Vh.dim()),
                                        lower=True)

        self.precision = np.dot(self.chol_inv.T, self.chol_inv)

        trial = dl.TrialFunction(Vh)
        test  = dl.TestFunction(Vh)
        
        domain_measure_inv = dl.Constant(1.0 \
                                / dl.assemble(dl.Constant(1.) * dl.dx(Vh.mesh())))

        #Identity mass matrix
        self.M = dl.assemble(domain_measure_inv * dl.inner(trial, test) * dl.dx)
        self.Msolver = Operator2Solver(self.M)

        if mean:
            self.mean = mean
        else:
            tmp = dl.Vector()
            self.M.init_vector(tmp, 0)
            tmp.zero()
            self.mean = tmp

        if Vh.dim() == 1:
            trial = dl.as_matrix([[trial]])
            test  = dl.as_matrix([[test]])

        #Create form matrices 
        covariance_op = dl.as_matrix(list(map(list, self.covariance)))
        precision_op  = dl.as_matrix(list(map(list, self.precision)))
        chol_op       = dl.as_matrix(list(map(list, self.chol)))
        chol_inv_op   = dl.as_matrix(list(map(list, self.chol_inv)))

        #variational for the regularization operator, or the precision matrix
        var_form_R = domain_measure_inv \
                     * dl.inner(test, dl.dot(precision_op, trial)) * dl.dx

        #variational for the inverse regularization operator, or the covariance
        #matrix
        var_form_Rinv = domain_measure_inv \
                        * dl.inner(test, dl.dot(covariance_op, trial)) * dl.dx

        #variational form for the square root of the regularization operator
        var_form_R_sqrt = domain_measure_inv \
                          * dl.inner(test, dl.dot(chol_inv_op.T, trial)) * dl.dx

        #variational form for the square root of the inverse regularization 
        #operator
        var_form_Rinv_sqrt = domain_measure_inv \
                             * dl.inner(test, dl.dot(chol_op, trial)) * dl.dx

        self.R         = dl.assemble(var_form_R)
        self.RSolverOp = dl.assemble(var_form_Rinv)
        self.Rsolver   = Operator2Solver(self.RSolverOp)
        self.sqrtR     = dl.assemble(var_form_R_sqrt)
        self.sqrtRinv  = dl.assemble(var_form_Rinv_sqrt)
        
    def init_vector(self, x, dim):
        """
        Inizialize a vector :code:`x` to be compatible with the 
        range/domain of :math:`R`.

        If :code:`dim == "noise"` inizialize :code:`x` to be compatible 
        with the size of white noise used for sampling.
        """

        if dim == "noise":
            self.sqrtRinv.init_vector(x, 1)
        else:
            self.sqrtRinv.init_vector(x, dim)

    def sample(self, noise, s, add_mean=True):
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a 
        sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
       
        self.sqrtRinv.mult(noise, s)

        if add_mean:
            s.axpy(1.0, self.mean)

