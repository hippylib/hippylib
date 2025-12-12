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

import dolfin as dl
import numpy as np


class TimeDependentVector:
    """
    A class to store time dependent vectors.
    Snapshots are stored/retrieved by specifying
    the time of the snapshot. Times at which the snapshot are taken must be
    specified in the constructor.
    """

    def __init__(self, times, tol=1e-10, mpi_comm=dl.MPI.comm_world):
        """
        Constructor:

        - :code:`times`: time frame at which snapshots are stored.
        - :code:`tol`  : tolerance to identify the frame of the snapshot.
        """
        self.nsteps = len(times)
        self.data = []

        for i in range(self.nsteps):
            self.data.append(dl.Vector(mpi_comm))

        self.times = times
        self.tol = tol
        self._mpi_comm = mpi_comm
        self.Vh = None


    def mpi_comm(self):
        """
        Return the MPI communicator associated to the vector.
        """
        return self._mpi_comm

    def copy(self):
        """
        Return a copy of all the time frames and snapshots
        """
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self._mpi_comm)
        res.Vh = self.Vh
        res.data = []

        for v in self.data:
            res.data.append(v.copy())

        return res

    def initialize(self, Vh):
        """
        Initialize all the snapshots to be compatible
        with the function space :code:`Vh`.
        """
        template_fun = dl.Function(Vh)
        self.data = []

        for i in range(self.nsteps):
            self.data.append(template_fun.vector().copy())

        self._mpi_comm = Vh.mesh().mpi_comm()  # update the communicator
        self.Vh = Vh  # store the function space

    def axpy(self, a, other):
        """
        Compute :math:`x = x + \\mbox{a*other}` snapshot per snapshot.
        """
        for i in range(self.nsteps):
            self.data[i].axpy(a, other.data[i])

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
        while i < self.nsteps - 1 and 2 * t > self.times[i] + self.times[i + 1]:
            i += 1

        assert abs(t - self.times[i]) < self.tol

        self.data[i].zero()
        self.data[i].axpy(1.0, u)

    def retrieve(self, u, t):
        """
        Retrieve snapshot :code:`u` relative to time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps - 1 and 2 * t > self.times[i] + self.times[i + 1]:
            i += 1

        assert abs(t - self.times[i]) < self.tol

        u.zero()
        u.axpy(1.0, self.data[i])

    def view(self, t):
        """
        Return a view of the snapshot at time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps - 1 and 2 * t > self.times[i] + self.times[i + 1]:
            i += 1

        assert abs(t - self.times[i]) < self.tol

        return self.data[i]

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
        assert (
            other.nsteps == self.nsteps
        ), "vectors do not have the same number of snapshots"
        a = 0.0
        for i in range(self.nsteps):
            a += self.data[i].inner(other.data[i])
        return a

    def element_wise_inner(self, other):
        """
        Compute the element-wise inner products: :math:`a[i] = (\\mbox{self[i]},\\mbox{other[i]})` for each snapshot.
        """
        assert (
            other.nsteps == self.nsteps
        ), "vectors do not have the same number of snapshots"
        out = np.zeros(self.nsteps)
        for i in range(self.nsteps):
            out[i] += self.data[i].inner(other.data[i])

        return out

    def matmul(self, mat, out):
        """
        Compute the matrix-vector product :math:`y = A*x` for each snapshot.
        """
        assert (
            mat.size(1) == self.data[0].size()
        ), f"Matrix and vector are not compatible. Matrix columns: {mat.size(1)} Vector size: {self.data[0].size()}."
        assert (
            self.nsteps == out.nsteps
        ), "time dependent vectors do not have the same number of snapshots"
        assert hasattr(mat, "mult"), "Matrix mat has no method: mult"

        for i in range(self.nsteps):
            mat.mult(self.data[i], out.data[i])

    def get_local(self):
        """
        Return the local part of the vector.
        """
        return np.array([v.get_local() for v in self.data]).flatten()

    def set_local(self, v):
        """
        Set the local part of the vector.
        """
        vv = np.reshape(v, (self.nsteps, -1))
        assert (
            vv.shape[0] == self.nsteps
        ), "The number of snapshots does not match the number of time steps"
        assert (
            vv.shape[1] == self.data[0].get_local().size
        ), "The size of the vector does not match the size of the snapshot"
        for i in range(self.nsteps):
            self.data[i].set_local(vv[i])
            self.data[i].apply("")


    def apply(self, mode):
        """
        Apply the changes to the vector.
        """
        for d in self.data:
            d.apply(mode)


    def gather_on_zero(self):
        """
        Gather the vector on process 0.
        """
        vec_size = self.data[0].size()

        if self._mpi_comm.rank == 0:
            vec_as_array = np.zeros((self.nsteps, vec_size))
        else:
            vec_as_array = np.array([])
        for i, d in enumerate(self.data):
            d_gathered = d.gather_on_zero()
            if self._mpi_comm.rank == 0:
                vec_as_array[i, :] = d_gathered

        vec = vec_as_array.flatten()
        return vec


    def __add__(self, other):
        if isinstance(other, TimeDependentVector):
            assert (
                other.nsteps == self.nsteps
            ), "vectors do not have the same number of snapshots"

            sum_vec = self.copy()
            sum_vec.axpy(1.0, other)

        elif isinstance(other, (int, float)):
            sum_vec = self.copy() 
            for d in sum_vec.data:
                d += other
        else:
            raise TypeError(f"Unsupported type {type(other)} for addition with TimeDependentVector")

        return sum_vec

    def __sub__(self, other):
        if isinstance(other, TimeDependentVector):
            assert (
                other.nsteps == self.nsteps
            ), "vectors do not have the same number of snapshots"

            diff_vec = self.copy()
            diff_vec.axpy(-1.0, other)

        elif isinstance(other, (int, float)):
            diff_vec = self.copy() 
            for d in diff_vec.data:
                d -= other
        else:
            raise TypeError(f"Unsupported type {type(other)} for subtraction with TimeDependentVector")

        return diff_vec


    def __mul__(self, other):
        if isinstance(other, (int, float)):
            prod_vec = self.copy() 
            for d in prod_vec.data:
                d *= other
        else:
            raise TypeError(f"Unsupported type {type(other)} for multiplication with TimeDependentVector")

        return prod_vec


    def __iadd__(self, other):
        if isinstance(other, TimeDependentVector):
            assert (
                other.nsteps == self.nsteps
            ), "vectors do not have the same number of snapshots"
            self.axpy(1.0, other)

        elif isinstance(other, (int, float)):
            for d in self.data:
                d += other
        else:
            raise TypeError(f"Unsupported type {type(other)} for addition with TimeDependentVector")

        return self


    def __isub__(self, other):
        if isinstance(other, TimeDependentVector):
            assert (
                other.nsteps == self.nsteps
            ), "vectors do not have the same number of snapshots"
            self.axpy(-1.0, other)

        elif isinstance(other, (int, float)):
            for d in self.data:
                d -= other
        else:
            raise TypeError(f"Unsupported type {type(other)} for subtraction with TimeDependentVector")

        return self


    def __imul__(self, other):
        if isinstance(other, (int, float)):
            for d in self.data:
                d *= other
        else:
            raise TypeError(f"Unsupported type {type(other)} for multiplication with TimeDependentVector")
        return self


    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        neg_vec = self.copy()
        for d in neg_vec.data:
            d *= -1
        return neg_vec
