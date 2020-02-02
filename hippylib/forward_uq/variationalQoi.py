# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
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
# Software Foundation) version 2.1 dated February 1999.

import dolfin as dl
import ufl
from ..modeling.variables import STATE, PARAMETER
from ..utils import experimental
from ..utils.vector2function import vector2Function
from .qoi import Qoi

@experimental(name = 'VariationalQoi',version='3.0.0', msg='Still need to work on handling Dirichlet boundary conditions for x[STATE]')
class VariationalQoi(Qoi):
    def __init__(self, Vh, qoi_varf):
        self.Vh = Vh
        self.qoi_varf = qoi_varf
        
        self.L = {}

    def eval(self, x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter a are accessed.
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        return dl.assemble(self.qoi_varf(u,m))
    
    def grad(self, i, x, g):
        if i == STATE:
            self.grad_state(x, g)
        elif i==PARAMETER:
            self.grad_param(x, g)
        else:
            raise i
                
    def grad_state(self,x,g):
        """Evaluate the gradient with respect to the state.
        Only the state u and (possibly) the parameter m are accessed. """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        g.zero()
        dl.assemble(dl.derivative(self.qoi_varf(u,m), u), tensor=g)
        
    def grad_param(self,x,g):
        """Evaluate the gradient with respect to the state.
        Only the state u and (possibly) the parameter m are accessed. """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        g.zero()
        dl.assemble(dl.derivative(self.qoi_varf(u,m), m), tensor=g)
                
    def apply_ij(self,i,j, dir, out):
        """Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the cost in direction dir."""
        if (i,j) in self.L:
            self.L[i,j].mult(dir, out)
        else:
            self.L[j,i].transpmult(dir,out)


    def setLinearizationPoint(self, x):
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        x = [u,m]
        for i in range(2):
            di_form = dl.derivative(self.qoi_varf(*x), x[i])
            for j in range(i,2):
                dij_form = dl.derivative(di_form,x[j] )
                self.L[i,j] = dl.assemble(dij_form)
