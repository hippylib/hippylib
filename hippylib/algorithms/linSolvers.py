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


def PETScKrylovSolver(comm, method, preconditioner):
    try:
        out = dl.PETScKrylovSolver(comm, method, preconditioner)
    except:
        out = dl.PETScKrylovSolver(method, preconditioner)
        
    return out

def _PETScLUSolver_set_operator(self, A):
    if hasattr(A, 'mat'):
        self.ksp().setOperators(A.mat())
    else:
        self.ksp().setOperators(dl.as_backend_type(A).mat())
    

def PETScLUSolver(comm, method='default'):
    if not hasattr(dl.PETScLUSolver, 'set_operator'):
        dl.PETScLUSolver.set_operator = _PETScLUSolver_set_operator

    return dl.PETScLUSolver(comm, method)

