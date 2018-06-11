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
from .tracers import NullTracer


class NullQoi(object):
    def __init__(self):
        pass
    def eval(self,x):
        return 0.



class SampleStruct:
    def __init__(self, kernel):
        self.derivative_info = kernel.derivativeInfo()
        self.u = kernel.model.generate_vector(STATE)
        self.m = kernel.model.generate_vector(PARAMETER)
        self.cost = 0
        
        if self.derivative_info >= 1:
            self.p  = kernel.model.generate_vector(STATE)
            self.g  = kernel.model.generate_vector(PARAMETER)
            self.Cg = kernel.model.generate_vector(PARAMETER)
        else:
            self.p = None
            self.g = None
        
    def assign(self, other):
        assert self.derivative_info == other.derivative_info
        self.cost = other.cost
        
        self.m = other.m.copy()
        self.u = other.u.copy()
        
        if self.derivative_info >= 1:
            self.g = other.g.copy()
            self.p = other.p.copy()
            self.Cg = other.Cg.copy()


class MCMC(object):
    def __init__(self, kernel):
        self.kernel = kernel
        self.parameters = {}
        self.parameters["number_of_samples"]     = 2000
        self.parameters["burn_in"]               = 1000
        self.parameters["print_progress"]        = 20
        self.parameters["print_level"]           = 1
        
        self.sum_q = 0.
        self.sum_q2 = 0.
        
    def run(self, m0, qoi=None, tracer = None):
        if qoi is None:
            qoi = NullQoi()
        if tracer is None:
            tracer = NullTracer()
        number_of_samples = self.parameters["number_of_samples"]
        burn_in = self.parameters["burn_in"]
        
        current = SampleStruct(self.kernel)
        proposed = SampleStruct(self.kernel)
        
        current.m.zero()
        current.m.axpy(1., m0)
        self.kernel.init_sample(current)
        
        if self.parameters["print_level"] > 0:
            print( "Burn {0} samples".format(burn_in) )
        sample_count = 0
        naccept = 0
        n_check = burn_in // self.parameters["print_progress"]
        while (sample_count < burn_in):
            naccept +=self.kernel.sample(current, proposed)
            sample_count += 1
            if sample_count % n_check == 0 and self.parameters["print_level"] > 0:
                print( "{0:2.1f} % completed, Acceptance ratio {1:2.1f} %".format(float(sample_count)/float(burn_in)*100,
                                                                         float(naccept)/float(sample_count)*100 ) )
        if self.parameters["print_level"] > 0:
            print( "Generate {0} samples".format(number_of_samples) )
        sample_count = 0
        naccept = 0
        n_check = number_of_samples // self.parameters["print_progress"]
        while (sample_count < number_of_samples):
            naccept +=self.kernel.sample(current, proposed)
            q = qoi.eval([current.u, current.m])
            self.sum_q += q
            self.sum_q2 += q*q
            tracer.append(current, q)
            sample_count += 1
            if sample_count % n_check == 0 and self.parameters["print_level"] > 0:
                print( "{0:2.1f} % completed, Acceptance ratio {1:2.1f} %".format(float(sample_count)/float(number_of_samples)*100,
                                                                         float(naccept)/float(sample_count)*100 ) )       
        return naccept
    
    def consume_random(self):
        number_of_samples = self.parameters["number_of_samples"]
        burn_in = self.parameters["burn_in"]
        
        for ii in range(number_of_samples+burn_in):
            self.kernel.consume_random()