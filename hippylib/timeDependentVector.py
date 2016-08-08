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

from dolfin import Vector
import numpy as np

class TimeDependentVector:
    """
    A class to store time dependent vectors.
    Snapshots are stored/retrieved by specifying
    the time of the snapshot.
    
    Times at which the snapshot are taken must be
    specified in the constructor.
    """
    
    def __init__(self, times, tol=1e-10):
        """
        Constructor:
        - times: time frame at which snapshots are stored
        - tol  : tolerance to identify the frame of the
                 snapshot.
        """
        self.nsteps = len(times)
        self.data = [];
        
        for i in range(self.nsteps):
            self.data.append( Vector() )
             
        self.times = times
        self.tol = tol
    
    def copy(self, other):
        """
        Copy all the time frames and snapshot from other to self.
        """
        self.nsteps = other.nsteps
        self.times = other.times
        self.tol = other.tol
        self.data = []
        
        for v in other.data:
            self.data.append( v.copy() )
        
    def initialize(self,M,dim):
        """
        Initialize all the snapshot to be compatible
        with the range/domain of an operator M.
        """
        
        for d in self.data:
            M.init_vector(d,dim)
            d.zero()
            
    def randn_perturb(self,std_dev):
        """
        Add a random perturbation eta_i ~ N(0, std_dev^2 I)
        to each snapshots.
        """
        for d in self.data:
            noise = std_dev * np.random.normal(0, 1, len(d.array()))
            d.set_local(d.array() + noise)
    
    def axpy(self, a, other):
        """
        Compute x = x + a*other snapshot per snapshot.
        """
        for i in range(self.nsteps):
            self.data[i].axpy(a,other.data[i])
        
    def zero(self):
        """
        Zero out each snapshot.
        """
        for d in self.data:
            d.zero()
            
    def store(self, u, t):
        """
        Store snapshot u relative to time t.
        If t does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps-1 and 2*t > self.times[i] + self.times[i+1]:
            i += 1
            
        assert abs(t - self.times[i]) < self.tol
        
        self.data[i].set_local( u.array() )
        
    def retrieve(self, u, t):
        """
        Retrieve snapshot u relative to time t.
        If t does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps-1 and 2*t > self.times[i] + self.times[i+1]:
            i += 1
            
        assert abs(t - self.times[i]) < self.tol
        
        u.set_local( self.data[i].array() )
        
    def norm(self, time_norm, space_norm):
        """
        Compute the space-time norm of the snapshot.
        """
        assert time_norm == "linf"
        s_norm = 0
        for i in range(self.nsteps):
            tmp = self.data[i].norm(space_norm)
            if tmp > s_norm:
                s_norm = tmp
        
        return s_norm
        
                