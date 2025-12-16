# Copyright (c) 2016-2018, The University of Texas at Austin
# & University of California--Merced.
# Copyright (c) 2019-2022, The University of Texas at Austin
# University of California--Merced, Washington University in St. Louis.
# Copyright (c) 2023-2025, The University of Texas at Austin
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

import unittest
import dolfin as dl
import numpy as np

import hippylib as hp


class TestTimeDependentVector(unittest.TestCase):
    """Test TimeDependentVector class"""

    def setUp(self):
        self.mesh = dl.UnitSquareMesh(12, 15)
        self.V = dl.FunctionSpace(self.mesh, "Lagrange", 1)
        self.times = [0.0, 1.0, 2.0]
        self.tdvector = hp.TimeDependentVector(self.times)
        self.tdvector.initialize(self.V)

        # load some data into the tdvector
        for i in range(self.tdvector.nsteps):
            tmp_fn = dl.interpolate(dl.Constant(i + 1), self.V)
            self.tdvector.data[i].axpy(1.0, tmp_fn.vector())

    def test_initialization(self):
        """Test initialization of TimeDependentVector"""
        template_fn = dl.Function(self.V)

        self.assertEqual(self.tdvector.times, self.times)
        self.assertEqual(len(self.tdvector.data), len(self.times))
        for tidx, t in enumerate(self.times):
            self.assertEqual(
                self.tdvector.data[tidx].size(), template_fn.vector().size()
            )
            self.assertEqual(self.tdvector.times[tidx], t)
            self.assertIsInstance(self.tdvector.data[tidx], dl.PETScVector)

    def test_data_access(self):
        """Test data access of TimeDependentVector"""
        tmp = dl.Function(self.V)
        for i in range(self.tdvector.nsteps):
            fni = dl.interpolate(dl.Constant(i + 1), self.V)
            self.tdvector.retrieve(tmp.vector(), self.times[i])
            assert np.allclose(fni.vector().get_local(), tmp.vector().get_local())
            assert np.allclose(
                self.tdvector.data[i].get_local(), tmp.vector().get_local()
            )

    def test_norm(self):
        """Test norm of TimeDependentVector"""
        TRUE = 1872.0
        assert np.isclose(self.tdvector.norm("linf", "l2"), np.sqrt(TRUE))

    def test_inner(self):
        """Test inner product of TimeDependentVector"""
        TRUE = 2912.0
        assert np.isclose(self.tdvector.inner(self.tdvector), TRUE)

    def test_elwise_inner(self):
        """Test element-wise inner product of TimeDependentVector"""
        TRUE = 208 * np.power([1.0, 2.0, 3.0], 2)
        assert np.allclose(self.tdvector.element_wise_inner(self.tdvector), TRUE)

    def test_get_local(self):
        """Test get_local of TimeDependentVector"""
        TRUE = np.array(
            [self.tdvector.data[i].get_local() for i in range(self.tdvector.nsteps)]
        ).flatten()
        assert np.allclose(self.tdvector.get_local(), TRUE)

    def test_set_local(self):
        """Test get_local of TimeDependentVector"""
        TRUE = []
        for i in range(self.tdvector.nsteps):
            TRUE.append(dl.interpolate(dl.Constant(i + 3), self.V).vector().get_local())

        self.tdvector.set_local(np.array(TRUE).flatten())

        assert np.allclose(self.tdvector.get_local(), np.array(TRUE).flatten())

        # check that the data got set correctly
        res = dl.Function(self.V)
        res.vector().zero()
        testfn = dl.Function(self.V)
        testfn.interpolate(dl.Constant(4))
        res.vector().axpy(1.0, testfn.vector())
        res.vector().axpy(-1.0, self.tdvector.data[1])

        assert (
            res.vector().norm("linf") < 1e-2
        ), "Functions not equal, potential MPI issue"

    def test_matmul(self):
        """Test matmul of TimeDependentVector"""
        mat = dl.assemble(dl.TestFunction(self.V) * dl.TrialFunction(self.V) * dl.dx)
        mat.zero()

        diag = dl.Function(self.V)
        diag.interpolate(dl.Constant(3.0))

        # new matrix should be 3*I
        mat.set_diagonal(diag.vector())

        out = self.tdvector.copy()
        out.zero()

        self.tdvector.matmul(mat, out)
        res = dl.Function(self.V)
        for i in range(self.tdvector.nsteps):
            res.vector().zero()
            testfn = dl.Function(self.V)
            testfn.interpolate(dl.Constant((i + 1) * 3.0))

            res.vector().axpy(1.0, testfn.vector())
            res.vector().axpy(-1.0, out.data[i])
            assert res.vector().norm("linf") < 1e-2
