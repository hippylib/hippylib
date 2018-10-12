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

import unittest
import dolfin as dl
import numpy as np
import os
from tempfile import NamedTemporaryFile


import sys
sys.path.append('../../../')
from hippylib import *

from .test_prior import TestPrior

class TestGaussianLRPosterior(unittest.TestCase):

    def setUp(self):

        mesh = dl.UnitSquareMesh(8, 8)
        Vh = dl.VectorFunctionSpace(mesh, 'R', 0, dim = 3)
        self.test_prior = TestPrior(Vh, variance=[1.0, 1.0, 1.0])

        m = dl.Vector()
        self.test_prior.init_vector(m, 0)
        self.test_prior.mean = m.copy()

        d1 = np.ones(3)
        U1 = MultiVector(self.test_prior.mean, 3)

        d2 = 2.0 * np.ones(3)
        U2 = MultiVector(self.test_prior.mean, 3)

        I = np.eye(3)
        for i in range(3):
            U1[i].set_local(I[:, i])
            U2[i].set_local(I[:, i])
        g1 = GaussianLRPosterior(self.test_prior, d1, U1)
        g1.mean = self.test_prior.mean.copy()

        g2 = GaussianLRPosterior(self.test_prior, d2, U2)
        g2.mean = self.test_prior.mean.copy()
        
        self.mix = GaussianLRPosteriorMixture(self.test_prior, [g1, g2], np.array([0.5, 0.5]))

    def tearDown(self):
        self.mix = None
        self.test_prior = None


    def test_log_prod_det(self):
        """
        Tests if the log of the product of determinants is being computed
        correctly
        """

        true_log_det_prod = [2.07944154167984, 3.29583686600433]
        self.assertTrue(np.allclose(true_log_det_prod, self.mix.log_det_prod),
                   msg = "Log of the product of determinants is incorrect.")


    def test_IS_ratio(self):
        """
        Tests if IS ratio, prior(m) / mix(m), is being computed
        correctly
        """

        actual_IS_ratio = self.mix.getISRatio(self.mix.prior.mean)
        true_IS_ratio   = 0.249234241890573

        self.assertAlmostEqual(
                actual_IS_ratio,
                true_IS_ratio,
                delta=1e-6)

    def test_IS_ratio_one(self):
        """
        IS ratio must be 1 if prior and posterior are equal
        """

        d = np.zeros(3)
        U = MultiVector(self.test_prior.mean, 3)

        I = np.eye(3)
        for i in range(3):
            U[i].set_local(I[:, i])
        g = GaussianLRPosterior(self.test_prior, d, U)
        g.mean = self.test_prior.mean.copy()
        
        mix = GaussianLRPosteriorMixture(self.test_prior, [g], np.array([1.0]))

        actual_IS_ratio = mix.getISRatio(mix.prior.mean)
        true_IS_ratio   = 1.0
        
        self.assertAlmostEqual(
                actual_IS_ratio,
                true_IS_ratio,
                delta=1e-6)


    def test_append_log_prod_det(self):
        """
        Tests if the log of the product of determinants is being computed
        correctly
        """

        d = np.array([1.0, 0.5, 0.25])
        U = MultiVector(self.test_prior.mean, 3)


        I = np.eye(3)
        for i in range(3):
            U[i].set_local(I[:, i])
        
        g = GaussianLRPosterior(self.test_prior, d, U)
        g.mean = self.test_prior.mean.copy()
        
        self.mix.append(g, np.array([0.25, 0.25, 0.5]))

        true_log_det_prod = [2.07944154167984, 
                             3.29583686600433,
                             1.32175583998232]

        error_msg = "Log of the product of determinants is incorrect after appending a new component."

        self.assertTrue(np.allclose(true_log_det_prod, self.mix.log_det_prod),
                        msg = error_msg )



    
    def test_append_IS_ratio(self):

        d = np.array([1.0, 0.5, 0.25])
        U = MultiVector(self.test_prior.mean, 3)


        I = np.eye(3)
        for i in range(3):
            U[i].set_local(I[:, i])
        
        g = GaussianLRPosterior(self.test_prior, d, U)
        g.mean = self.test_prior.mean.copy()
        
        self.mix.append(g, np.array([0.25, 0.25, 0.5]))
        
        actual_IS_ratio = self.mix.getISRatio(self.mix.prior.mean)
        true_IS_ratio   = 0.336203307833023

        self.assertAlmostEqual(
                actual_IS_ratio,
                true_IS_ratio,
                delta=1e-6)

