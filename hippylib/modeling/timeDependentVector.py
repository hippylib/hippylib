# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
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
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl

class TimeDependentVector(object):
    """
    A class to store time dependent vectors.
    Snapshots are stored/retrieved by specifying
    the time of the snapshot. Times at which the snapshot are taken must be
    specified in the constructor.
    """
    
    def __init__(self, times, tol=1e-10, mpi_comm = dl.MPI.comm_world):
        """
        Constructor:

        - :code:`times`: time frame at which snapshots are stored.
        - :code:`tol`  : tolerance to identify the frame of the snapshot.
        """
        self.nsteps = len(times)
        self.data = []
        
        for i in range(self.nsteps):
            self.data.append( dl.Vector(mpi_comm) )
             
        self.times = times
        self.tol = tol
        self.mpi_comm = mpi_comm

    def __imul__(self, other):
        for d in self.data:
            d *= other
        return self
    

    def copy(self):
        """
        Return a copy of all the time frames and snapshots
        """        
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self.mpi_comm)
        res.data = []

        for v in self.data:
            res.data.append(v.copy())

        return res
        
    def initialize(self,M,dim):
        """
        Initialize all the snapshot to be compatible
        with the range/domain of an operator :code:`M`.
        """
        
        for d in self.data:
            M.init_vector(d,dim)
            d.zero()
            
    def axpy(self, a, other):
        """
        Compute :math:`x = x + \\mbox{a*other}` snapshot per snapshot.
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
        Store snapshot :code:`u` relative to time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps-1 and 2*t > self.times[i] + self.times[i+1]:
            i += 1
            
        assert abs(t - self.times[i]) < self.tol
        
        self.data[i].zero()
        self.data[i].axpy(1., u )
        
    def retrieve(self, u, t):
        """
        Retrieve snapshot :code:`u` relative to time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps-1 and 2*t > self.times[i] + self.times[i+1]:
            i += 1
            
        assert abs(t - self.times[i]) < self.tol
        
        u.zero()
        u.axpy(1., self.data[i] )
        
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
        
    def inner(self, other):
        """
        Compute the inner products: :math:`a+= (\\mbox{self[i]},\\mbox{other[i]})` for each snapshot.
        """
        a = 0.
        for i in range(self.nsteps):
            a += self.data[i].inner(other.data[i])
        return a
 

