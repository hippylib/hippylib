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

from ..modeling.variables import STATE, PARAMETER
from ..utils.vector2function import vector2Function

import numpy as np


class NullTracer(object):
    def __init__(self):
        pass
    def append(self,current, q):
        pass
    
class QoiTracer(object):
    def __init__(self, n):
        self.data = np.zeros(n)
        self.i = 0
        
    def append(self,current, q):
        self.data[self.i] = q
        self.i+=1
        
class FullTracer(object):
    def __init__(self, n, Vh, par_fid = None, state_fid = None):
        self.data = np.zeros((n,2))
        self.i = 0
        self.Vh = Vh
        self.par_fid = par_fid
        self.state_fid = state_fid
        
    def append(self,current, q):
        self.data[self.i, 0] = q
        self.data[self.i, 1] = current.cost
        if self.par_fid is not None:
            self.par_fid << vector2Function(current.m, self.Vh[PARAMETER], name="parameter")
        if self.state_fid is not None:
            self.state_fid << vector2Function(current.u, self.Vh[STATE], name = "state")
        self.i+=1