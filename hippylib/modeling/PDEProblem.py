# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2022, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
# Copyright (c) 2023-2024, The University of Texas at Austin 
# & University of California--Merced.
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

class PDEProblem(object):
    """ Consider the PDE problem:
        Given :math:`m`, find :math:`u` such that 
        
            .. math:: F(u, m, p) = ( f(u, m), p) = 0, \\quad \\forall p.
        
        Here :math:`F` is linear in :math:`p`, but it may be non linear in :math:`u` and :math:`m`.
    """

    def generate_state(self):
        """ Return a vector in the shape of the state. """
        raise NotImplementedError("Child class should implement method generate_state")

    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        raise NotImplementedError("Child class should implement method generate_parameter")

    def init_parameter(self, m):
        """ Initialize the parameter. """
        raise NotImplementedError("Child class should implement method init_parameter")

    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that

            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0, \\quad \\forall \\hat{p}.
        """
        raise NotImplementedError("Child class should implement method solveFwd")

    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear adjoint problem: 
            Given :math:`m`, :math:`u`; find :math:`p` such that
            
                .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        raise NotImplementedError("Child class should implement method solveAdj")

    def evalGradientParameter(self, x, out):
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        raise NotImplementedError("Child class should implement method evalGradientParameter")
 
    def setLinearizationPoint(self,x, gauss_newton_approx):

        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. 
            Set whether Gauss Newton approximation of
            the Hessian should be used."""
        raise NotImplementedError("Child class should implement method setLinearizationPoint")
      
    def solveIncremental(self, out, rhs, is_adj):
        """ If :code:`is_adj = False`:

            Solve the forward incremental system:
            Given :math:`u, m`, find :math:`\\tilde{u}` such that

            .. math::
                \\delta_{pu} F(u, m, p; \\hat{p}, \\tilde{u}) = \\mbox{rhs}, \\quad \\forall \\hat{p}.
            
            If :code:`is_adj = True`:
            
            Solve the adjoint incremental system:
            Given :math:`u, m`, find :math:`\\tilde{p}` such that

            .. math::
                \\delta_{up} F(u, m, p; \\hat{u}, \\tilde{p}) = \\mbox{rhs}, \\quad \\forall \\hat{u}.
        """
        raise NotImplementedError("Child class should implement method solveIncremental")

    def apply_ij(self,i,j, dir, out):   
        """
            Given :math:`u, m, p`; compute 
            :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`, 
            :math:`\\forall \\hat{i}.`
        """
        raise NotImplementedError("Child class should implement method apply_ij")
        
    def apply_ijk(self,i,j,k, x, jdir, kdir, out):
        """
            Given :code:`x = [u,a,p]`; compute
            :math:`\\delta_{ijk} F(u,a,p; \\hat{i}, \\tilde{j}, \\tilde{k})`
            in the direction :math:`(\\tilde{j},\\tilde{k}) = (`:code:`jdir,kdir`), :math:`\\forall \\hat{i}.`
        """
        raise NotImplementedError("Child class should implement apply_ijk")
