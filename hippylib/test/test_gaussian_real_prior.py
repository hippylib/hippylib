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
import scipy.stats as scistat

from numpy.testing import assert_allclose

import sys
sys.path.append('../../')
from hippylib import *

def get_prior_sample(pr):

    noise = dl.Vector()
    pr.init_vector(noise, "noise")
    parRandom.normal(1.0, noise)
    
    s = dl.Vector()
    pr.init_vector(s, 0)
    pr.sample(noise, s, add_mean=True)
    
    return s

class TestGaussianRealPrior(unittest.TestCase):
    """
    Test suite for prior.GaussianRealPrior
    """

    def setUp(self):
        np.random.seed(1)
        #self.dim = np.random.randint(1, high=5)
        self.dim = 1

        self.means = np.random.uniform(-10, high=10., size=self.dim)

        self.chol = np.tril(np.random.uniform(1, high=10, size=(self.dim,self.dim)))

        self.cov = np.dot(self.chol, self.chol.T)

        self.precision = np.linalg.inv(self.cov)

        mesh = dl.RectangleMesh(dl.mpi_comm_world(), 
                                dl.Point(0.0, 0.0),
                                dl.Point(3,2), 6, 4)

        if self.dim > 1:
            self.Rn = dl.VectorFunctionSpace(mesh, "R", 0, dim=self.dim)
        else:
            self.Rn = dl.FunctionSpace(mesh, "R", 0)

        self.test_prior = GaussianRealPrior(self.Rn, self.cov)
        

        m = dl.Function(self.Rn)
        m.vector().zero()
        m.vector().set_local(self.means)
        self.test_prior.mean.axpy(1., m.vector())

    
    def tearDown(self):

        self.test_prior = None

    def test_incompatible_cov(self):
        #Test whether using an incompatible covariance matrix results in
        #ValueError

        incompatible_cov = np.identity(self.dim + 1)

        with self.assertRaises(ValueError):
            temp = GaussianRealPrior(self.Rn, incompatible_cov)

    def test_int_cov(self):
        #Test whether using an integer covariance matrix results in
        #TypeError

        int_cov = np.identity(self.dim).astype(np.int64)

        with self.assertRaises(TypeError):
            temp = GaussianRealPrior(self.Rn, int_cov)


    def test_cost_at_mean(self):
        #Test whether cost at mean is 0 or not

        expected_cost = 0.0
        actual_cost = self.test_prior.cost(self.test_prior.mean)

        self.assertAlmostEqual(
                expected_cost,
                actual_cost,
                delta=1e-10)

    def test_cost_sample(self):
        #Test cost at some random vector

        sample = get_prior_sample(self.test_prior)

        diff = sample.get_local() - self.means

        expected_cost = 0.5 * np.inner(diff, np.dot(self.precision, diff))

        actual_cost = self.test_prior.cost(sample)

        self.assertAlmostEqual(
                    expected_cost, 
                    actual_cost, 
                    delta=1e-10)

    def test_sample_mean(self):
        #Ensure sample mean converges to true mean

        n_samples = 10000

        samples = [get_prior_sample(self.test_prior).get_local() \
                   for i in range(n_samples)]
               
        sample_mean = np.mean(samples, axis=0)

        rel_err = np.linalg.norm(sample_mean - self.means) \
                  / np.linalg.norm(self.means)
                
        assert_allclose(
                    self.means,
                    sample_mean, 
                    rtol=0.1)

    def test_sample_cov(self):
        #Ensure sample cov converges to true cov

        n_samples = 10000

        samples = [get_prior_sample(self.test_prior).get_local() \
                   for i in range(n_samples)]
       
        samples = np.array(samples)
        sample_cov = np.cov(samples.T)

        rel_err = np.linalg.norm(sample_cov - self.test_prior.covariance) \
                  / np.linalg.norm(self.test_prior.covariance)
        
        print('Actual:', sample_cov)
        print('Expected:', self.test_prior.covariance)

        self.assertAlmostEqual(
                rel_err,
                0.0,
                delta=0.1)
 
    
    def test_trace(self):
        #Ensure covariance trace is correct
        expected_trace = np.sum(np.diag(self.test_prior.covariance))        

        tol = 1e-2

        actual_trace = self.test_prior.trace(tol=tol)

        self.assertAlmostEqual(
                expected_trace,
                actual_trace,
                delta=tol)

    def test_grad(self):
        #Ensure cost gradient is correct

        sample = get_prior_sample(self.test_prior)

        actual_grad = dl.Vector()
        self.test_prior.init_vector(actual_grad, 0)
        self.test_prior.grad(sample, actual_grad)

        expected_grad_np = np.dot(self.precision, 
                                  sample.get_local() - self.means)

        self.assertTrue(
                np.allclose(expected_grad_np, actual_grad.get_local()))

    def test_Rinv_sqrt(self):
        #Ensures that the assembled sqrtRinv matrix is the cholesky factor of
        #the covariance matrix


        self.assertTrue(
                np.allclose(
                    self.test_prior.sqrtRinv.array(),
                                    self.test_prior.chol))

    
    def test_R_sqrt(self):
        #Ensures that the assembled sqrtR matrix is the cholesky factor of
        #the precision matrix

        self.assertTrue(
                np.allclose(
                    self.test_prior.sqrtR.array(),
                                    self.test_prior.chol_inv.T))
 
    def test_precision_matrix(self):
        #Ensures that the assembled precision matrix is equal to the supplied
        #precision matrix

        self.assertTrue(
                np.allclose(
                    self.test_prior.R.array(),
                    self.test_prior.precision))
 
    def test_covariance_matrix(self):
        #Ensures that the assembled precision matrix is equal to the supplied
        #precision matrix

        self.assertTrue(
                np.allclose(
                    self.test_prior.RSolverOp.array(),
                    self.test_prior.covariance))
    

if __name__ == '__main__':
    unittest.main()

