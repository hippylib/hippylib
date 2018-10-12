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
import numpy as np
import sys
sys.path.append('../../../')

from hippylib.forward_uq.rare_events.optLikeVar import getTruncStats, \
                                                       _trunc_helper1, \
                                                       _trunc_helper2

class TestTruncatedNormalsStats(unittest.TestCase):
    
    def test_trunc_mean(self):
        # mean of the truncated normal must remain the same as the original
        # distribution if truncation is symmetric

        y_min    = np.random.uniform()
        y_max    = np.random.uniform()
        avg      = 0.5 * (y_min + y_max)
        variance = 3.0


        [_, actual_avg, _] = getTruncStats(avg, variance, y_min, y_max)
        expected_avg       = avg

        self.assertEqual(actual_avg, expected_avg)


    def test_inf_limits(self):

        #Test moments of a truncated normal distribution if truncation limits
        #are [-inf, inf]. With these limits, there is no truncation and moments
        #should be the same as that of the original normal distribution


        avg      = 2.0
        variance = 4.0
        y_min    = -1e12
        y_max    = 1e12

        [trunc_mu, trunc_avg, trunc_var] = getTruncStats(\
                                            avg, variance, y_min, y_max)

        actual_output   = np.array([trunc_mu, trunc_avg, trunc_var])
        expected_output = np.array([1.0, avg, variance])

        self.assertTrue(np.allclose(actual_output, expected_output))

       
    def test_trunc_stats_values(self):

        # Reference:
        # https://github.com/cossio/TruncatedNormal.jl/blob/master/test/moments.jl

        avg      = -2.0
        variance = 9.0
        y_min    = 50.0
        y_max    = 70.0


        [_, actual_avg, actual_var] = getTruncStats(avg, variance, y_min, y_max)
        
        expected_avg = 50.1719434998988
        expected_var = 0.0293734381071683

        actual   = np.array([actual_avg, actual_var])
        expected = np.array([expected_avg, expected_var])

        self.assertTrue(np.allclose(actual, expected))

    def test_trunc_helper_function1(self):

        actual = np.array(\
                  [_trunc_helper1( 1.0,  1.0 + 1e-8),
                  _trunc_helper1(  0.5,  0.5 ), 
                  _trunc_helper1( -2.0,  1.0 ), 
                  _trunc_helper1( -2.0, -1.0 ), 
                  _trunc_helper1(  1.0,  2.0 )])
        
        expected = np.array([1.7724538597677852522848499, 
                             0.5 * np.sqrt(np.pi), 
                             -0.190184666491092019086029, 
                             -2.290397265491751547564988, 
                             2.29039726549175154756498826])
        
        self.assertTrue(np.allclose(actual, expected))


    def test_trunc_helper_function_2(self):

        actual = np.array(
                [_trunc_helper2(  1.0,  1.0 + 1e-8 ), 
                 _trunc_helper2(  0.5,  0.5  ), 
                 _trunc_helper2( -1.0,  1.0  ), 
                 _trunc_helper2( -2.0, -1.0  ), 
                 _trunc_helper2(  1.0,  2.0  )])
        
        expected = np.array( [0.886226943177296522704243, 
                              0.5 * 0.5 * np.sqrt(np.pi) - np.sqrt(np.pi) / 2, 
                             -0.436548113220292413, 
                              2.17039030552464315391802, 
                              2.170390305524643153918] )

        self.assertTrue(np.allclose(actual, expected))

 

