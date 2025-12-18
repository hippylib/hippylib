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

        self.true_scalar_sum = hp.TimeDependentVector(self.times)
        self.true_scalar_sum.initialize(self.V)

        self.true_scalar_sub = hp.TimeDependentVector(self.times)
        self.true_scalar_sub.initialize(self.V)

        self.true_scalar_mul = hp.TimeDependentVector(self.times)
        self.true_scalar_mul.initialize(self.V)

        self.scalar_for_mul = 3.0
        self.scalar_for_add = 3.0
        self.scalar_for_sub = 3.0

        # load some data into the tdvector
        for i in range(self.tdvector.nsteps):
            tmp_fn = dl.interpolate(dl.Constant(i + 1), self.V)
            self.tdvector.data[i].axpy(1.0, tmp_fn.vector())

            tmp_fn = dl.interpolate(dl.Constant(self.scalar_for_add + i + 1), self.V)
            self.true_scalar_sum.data[i].axpy(1.0, tmp_fn.vector())

            tmp_fn = dl.interpolate(dl.Constant(i + 1 - self.scalar_for_sub), self.V)
            self.true_scalar_sub.data[i].axpy(1.0, tmp_fn.vector())

            tmp_fn = dl.interpolate(dl.Constant((i + 1) * self.scalar_for_mul), self.V)
            self.true_scalar_mul.data[i].axpy(1.0, tmp_fn.vector())





    

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

    def test_add_with_tdvec(self):
        """Test addition of TimeDependentVector"""

        true_sum = hp.TimeDependentVector(self.times)
        true_sum.initialize(self.V)
        true_sum.zero()
        true_sum.axpy(2.0, self.tdvector)

        tdvec_sum = self.tdvector + self.tdvector

        # First check self.tdvector is unchanged
        diff_with_original = tdvec_sum.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its sum are the same"


        diff = true_sum.copy()
        diff.axpy(-1.0, tdvec_sum)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in add of two TimeDependentVector: {error_norm}")
        assert error_norm <= 1e-8, "Sum of two TimeDependentVector is incorrect"



    def test_sub_with_tdvec(self):
        """Test addition of TimeDependentVector"""

        tdvec_diff = self.tdvector - self.tdvector

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_diff.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its difference are the same"

        # Check that the difference is as expected (i.e., zero)
        error_norm = tdvec_diff.norm("linf", "l2")
        # print(f"Error norm in sub of two TimeDependentVector: {error_norm}")
        assert error_norm <= 1e-8, "Difference of two TimeDependentVector is incorrect"


    def test_mul_with_scalar(self):
        """Test multiplication of TimeDependentVector with scalar"""

        # Test one order of multiplication
        tdvec_prod = self.tdvector * self.scalar_for_mul

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_prod.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its product are the same"

        # Check multiplication error
        diff = self.true_scalar_mul.copy()
        diff.axpy(-1.0, tdvec_prod)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in mul of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8

        # Test the other way around
        tdvec_prod = self.scalar_for_mul * self.tdvector

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_prod.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its product are the same"

        # Check multiplication error
        diff = self.true_scalar_mul.copy()
        diff.axpy(-1.0, tdvec_prod)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in rmul of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8


    def test_add_with_scalar(self):
        """Test addition of TimeDependentVector with scalar"""
        # Test sum one way 

        tdvec_sum = self.tdvector + self.scalar_for_add

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_sum.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its sum are the same"

        # Check addition error
        diff = self.true_scalar_sum.copy()
        diff.axpy(-1.0, tdvec_sum)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in add of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8

        # Sum the other way
        tdvec_sum = self.scalar_for_add + self.tdvector

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_sum.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its sum are the same"

        # Check addition error
        diff = self.true_scalar_sum.copy()
        diff.axpy(-1.0, tdvec_sum)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in radd of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8


    def test_sub_with_scalar(self):
        """Test addition of TimeDependentVector with scalar"""

        # Subtract one way 
        tdvec_sub = self.tdvector - self.scalar_for_sub

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_sub.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its subtraction are the same"

        # Check subtraction error 
        diff = self.true_scalar_sub.copy()
        diff.axpy(-1.0, tdvec_sub)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in add of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8


        # Substract the other way 
        tdvec_sub = self.scalar_for_sub - self.tdvector

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_sub.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its subtraction are the same"

        # Check subtraction error 
        diff = self.true_scalar_sub.copy()
        diff.axpy(-1.0, tdvec_sub)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in add of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8


    def test_negation(self):
        tdvector_neg = -self.tdvector
        true_neg = hp.TimeDependentVector(self.times)
        true_neg.initialize(self.V)
        true_neg.zero()
        true_neg.axpy(-1.0, self.tdvector)

        # check that self.tdvector is unchanged
        diff_with_original = tdvector_neg.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its negation are the same"

        # Negation error 
        diff = true_neg.copy()
        diff.axpy(-1.0, tdvector_neg)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in negation of TimeDependentVector: {error_norm}")
        assert error_norm <= 1e-8, "Negation of TimeDependentVector is incorrect"


    def test_gather_on_zero(self):
        """Test gather_on_zero of TimeDependentVector"""

        gathered = self.tdvector.gather_on_zero()

        if self.tdvector.mpi_comm().rank == 0:
            vec_size = self.tdvector.data[0].size()
            nsteps = self.tdvector.nsteps
            assert gathered.shape[0] == vec_size * nsteps

            for i in range(nsteps):
                # slice for timestep 
                start_index = i * vec_size
                end_index = (i + 1) * vec_size
                time_vec = gathered[start_index:end_index]

                # True value should be i + 1 everywhere based on initialization
                assert np.allclose(time_vec, i+1)

    def test_iadd_with_tdvec(self):
        """Test in-place addition of TimeDependentVector"""

        tdvec_copy = self.tdvector.copy()
        tdvec_copy += self.tdvector

        true_sum = hp.TimeDependentVector(self.times)
        true_sum.initialize(self.V)
        true_sum.zero()
        true_sum.axpy(2.0, self.tdvector)

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_copy.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its sum are the same"

        diff = true_sum.copy()
        diff.axpy(-1.0, tdvec_copy)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in in-place add of two TimeDependentVector: {error_norm}")
        assert error_norm <= 1e-8, "In-place sum of two TimeDependentVector is incorrect"

    def test_isub_with_tdvec(self):
        """Test in-place subtraction of TimeDependentVector"""

        tdvec_copy = self.tdvector.copy()
        tdvec_copy -= self.tdvector

        # Check that self.tdvector is unchanged
        diff_with_original = tdvec_copy.copy()
        diff_with_original.axpy(-1.0, self.tdvector)
        diff_with_original_norm = diff_with_original.norm("linf", "l2")
        assert diff_with_original_norm > 1e-2, "self.tdvector and its difference are the same"

        # Check that the difference is as expected (i.e., zero)
        error_norm = tdvec_copy.norm("linf", "l2")
        # print(f"Error norm in in-place sub of two TimeDependentVector: {error_norm}")
        assert error_norm <= 1e-8, "In-place difference of two TimeDependentVector is incorrect"

    def test_iadd_with_scalar(self):
        """Test in-place addition of TimeDependentVector with scalar"""

        tdvec_copy = self.tdvector.copy()
        tdvec_copy += self.scalar_for_add

        # Check addition error
        diff = self.true_scalar_sum.copy()
        diff.axpy(-1.0, tdvec_copy)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in in-place add of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8, "In-place sum of TimeDependentVector with scalar is incorrect"


    def test_isub_with_scalar(self):
        """Test in-place subtraction of TimeDependentVector with scalar"""

        tdvec_copy = self.tdvector.copy()
        tdvec_copy -= self.scalar_for_sub

        # Check subtraction error 
        diff = self.true_scalar_sub.copy()
        diff.axpy(-1.0, tdvec_copy)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in in-place sub of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8, "In-place difference of TimeDependentVector with scalar is incorrect"

    def test_imul_with_scalar(self):    
        """Test in-place multiplication of TimeDependentVector with scalar"""

        tdvec_copy = self.tdvector.copy()
        tdvec_copy *= self.scalar_for_mul

        # Check multiplication error
        diff = self.true_scalar_mul.copy()
        diff.axpy(-1.0, tdvec_copy)
        error_norm = diff.norm("linf", "l2")
        # print(f"Error norm in in-place mul of TimeDependentVector with scalar: {error_norm}")
        assert error_norm <= 1e-8, "In-place multiplication of TimeDependentVector with scalar is incorrect"

if __name__ == '__main__':
    unittest.main()
