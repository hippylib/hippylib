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
import warnings

import sys
sys.path.append('../../')
from hippylib import deprecated, hIPPYlibDeprecationWarning

@deprecated(version='2.2.0', msg='Blah')
def blabla(a):
    print(a)

@deprecated(name='blu', version='2.2.0', msg='Blu')
def blu1(a):
    print('blu:', a)
    

class TestDeprecationWarnings(unittest.TestCase):
    """
    Test suite for deprecation warnings
    """
    def test_deprecated_no_name(self):
        with warnings.catch_warnings(record=True) as w:
            blabla(1)
            self.assertTrue(len(w) == 1)
            expected = "WARNING: {0} DEPRECATED since v{1}. {2}".format('blabla', '2.2.0', 'Blah')
            self.assertTrue(expected in str(w[0].message) )
            
    def test_deprecated_with_name(self):
        with warnings.catch_warnings(record=True) as w:
            blu1(1)
            self.assertTrue(len(w) == 1)
            expected = "WARNING: {0} DEPRECATED since v{1}. {2}".format('blu', '2.2.0', 'Blu')
            self.assertTrue(expected in str(w[0].message))
            
if __name__ == '__main__':
    unittest.main()


