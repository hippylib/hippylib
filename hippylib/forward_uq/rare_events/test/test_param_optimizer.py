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
from unittest.mock import Mock
import numpy as np


import sys
sys.path.append('../../../')
from hippylib import *


class TestParamsOptimizer(unittest.TestCase):

    def test_kl_div(self):
        # kl div must be 0 when the ideal and 
        # the target IS distributions are identical

        mock_lin_approx = Mock()
        instance = mock_lin_approx()

        instance.lin_std_dev        = 2.0 * np.random.uniform()
        instance.eval.return_value = 2.0 * np.random.uniform()

        param_opt = ParamOptimizer([-1e16, 1e16], instance)

        actual_kl_div = param_opt.getKLDist([1e16, 1.0])
        expected_kl_div = 0.0

        self.assertAlmostEqual(actual_kl_div, expected_kl_div, delta=1e-10)

if __name__ == '__main__':
    unittest.main()
