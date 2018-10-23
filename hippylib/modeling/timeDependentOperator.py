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

from .timeDependentVector import TimeDependentVector
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linalg import Transpose 
from ..utils.vector2function import vector2Function
from ..utils.checkDolfinVersion import dlversion


class TimeDependentOperator(object):
    """
    Operator for TimeDependentVector
    Single or multiple operators of same size are allowed
    """
    def __init__(self, A = None):
        if A != None:
            self.operator = A
            self.single_operator = True
        else:
            self.operator = []
            self.single_operator = False

    def zero(self):
        if self.single_operator:
            self.operator.zero()
        else:  
            try: 
                [op.zero for op in self.operator]
            except:
                pass

    def init_vector(self, x, dim):
        if self.single_operator:
            x.initialize(self.operator, dim)
        else:
            x.initialize(self.operator[0], dim)

    def mult(self, x, y, transp = False):
        if isinstance(x, TimeDependentVector) and isinstance(y, TimeDependentVector):
            self._mult_TD_TD(x, y, transp = transp)
        elif isinstance(x, TimeDependentVector) and (isinstance(y, dl.Vector) or isinstance(y, dl.GenericVector)):
            self._mult_TD_vec(x, y, transp = transp)
        elif (isinstance(x, dl.Vector) or isinstance(x, dl.GenericVector)) and isinstance(y, TimeDependentVector):
            self._mult_vec_TD(x, y, transp = transp)
        else:
            raise Exception("input of TimeDependentOperator not supported")

    def _mult_TD_TD(self, x, y, transp = False):
        xt = x.data[0].copy()
        yt = y.data[0].copy()
        if self.single_operator:
            if transp == False:
                for t in x.times:
                    x.retrieve(xt, t)
                    self.operator.mult(xt, yt)
                    y.store(yt, t)
            else:
                for t in x.times:
                    x.retrieve(xt, t)
                    self.operator.transpmult(xt, yt)
                    y.store(yt, t)
        else: 
            if transp == False:
                for i, t in enumerate(x.times):
                    x.retrieve(xt, t)
                    self.operator[i].mult(xt, yt)
                    y.store(yt, t)
            else:
                for i, t in enumerate(x.times):
                    x.retrieve(xt, t)
                    self.operator[i].transpmult(xt, yt)
                    y.store(yt, t)

    def _mult_vec_TD(self, x, y, transp = False):
        yt = y.data[0].copy()
        if self.single_operator:
            if transp == False:
                for t in y.times:
                    self.operator.mult(x, yt)
                    y.store(yt, t)
            else:
                for t in y.times:
                    self.operator.transpmult(x, yt)
                    y.store(yt, t)
        else: 
            if transp == False:
                for i, t in enumerate(y.times):
                    self.operator[i].mult(x, yt)
                    y.store(yt, t)
            else:
                for i, t in enumerate(y.times):
                    self.operator[i].transpmult(x, yt)
                    y.store(yt, t)

    def _mult_TD_vec(self, x, y, transp = False):
        xt = x.data[0].copy()
        y.zero()
        help1 = y.copy()
        if self.single_operator:
            if transp == False:
                for t in x.times:
                    x.retrieve(xt, t)
                    self.operator.mult(xt, help1)
                    y.axpy(1., help1)
            else:
                for t in x.times:
                    x.retrieve(xt, t)
                    self.operator.transpmult(xt, help1)
                    y.axpy(1., help1)
        else: 
            if transp == False:
                for i, t in enumerate(x.times):
                    x.retrieve(xt, t)
                    self.operator[i].mult(xt, help1)
                    y.axpy(1., help1)
            else:
                for i, t in enumerate(x.times):
                    x.retrieve(xt, t)
                    self.operator[i].transpmult(xt, help1)
                    y.axpy(1., help1)

    def transpmult(self, x, y):
        self.mult(x, y, transp = True)
