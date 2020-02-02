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

import unittest 
import warnings

import sys
sys.path.append('../../')
from hippylib import experimental

@experimental(version='2.2.0', msg='Blah')
def blabla(a):
    print(a)

@experimental(name='blu', version='2.2.0', msg='Blu')
def blu1(a):
    print('blu:', a)
    

class TestExperimentalWarnings(unittest.TestCase):
    """
    Test suite for deprecation warnings
    """
    def test_experimental_no_name(self):
        with warnings.catch_warnings(record=True) as w:
            blabla(1)
            self.assertTrue(len(w) == 1)
            expected = "WARNING: {0}  is an experimental function in v{1}. {2}".format('blabla', '2.2.0', 'Blah')
            self.assertTrue(expected in str(w[0].message) )
            
    def test_deprecated_with_name(self):
        with warnings.catch_warnings(record=True) as w:
            blu1(1)
            self.assertTrue(len(w) == 1)
            expected = "WARNING: {0}  is an experimental function in v{1}. {2}".format('blu', '2.2.0', 'Blu')
            self.assertTrue(expected in str(w[0].message))
            
if __name__ == '__main__':
    import dolfin as dl
    mpi_comm_world = dl.MPI.comm_world
    mpi_size = dl.MPI.size(mpi_comm_world)
    if mpi_size == 1:
        unittest.main()


