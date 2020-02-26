# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
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
import numpy as np
from numpy.testing import assert_allclose
from mpi4py import MPI

def splitCommunicators(comm_world, n_subdomain, n_instances):
    mpi_rank = comm_world.rank
    world_size = comm_world.size
    assert world_size == n_subdomain*n_instances
    # This interprets the communication structure as a grid where rows correspond
    # to instances and columns correspond to mesh subdomain collectives accross 
    # different samples. Color corresponds to row index, and key corresponds to 
    # column index as is customary. 
    color = np.floor(mpi_rank/n_instances)
    key = np.remainder(mpi_rank) 
    mesh_constructor_comm = comm_world.Split(color = color,key = key)
    collective_comm = comm_world.Split(color = key,key = color)
    return mesh_constructor_comm, collective_comm



def checkConsistentPartitioning(mesh, collective):
    DG0 = dl.FunctionSpace(mesh,'DG',0)
    CG1 = dl.FunctionSpace(mesh,'CG',1)
    collective_rank = collective.rank
    DG0_v = dl.interpolate(dl.Constant(float(collective_rank)),DG0)
    CG1_v = dl.interpolate(dl.Constant(float(collective_rank)),CG1)

    if collective_rank == 0:
        root_DG0_v = dl.interpolate(dl.Constant(float(collective_rank)),DG0)
        root_CG1_v = dl.interpolate(dl.Constant(float(collective_rank)),CG1)
    else:
        root_DG0_v = dl.interpolate(dl.Constant(0.),DG0)
        root_CG1_v = dl.interpolate(dl.Constant(0.),CG1)

    root_DG0_v.vector().set_local(collective.bcast(root_DG0_v.vector(),root = 0))
    root_CG1_v.vector().set_local(collective.bcast(root_DG0_v.vector(),root = 0))

    diff_DG0 = DG0_v - root_DG0_v
    diff_CG1 = CG1_v - root_CG1_v

    assert_allclose( [diff_DG0.norm("l2")], [0.])
    assert_allclose( [diff_CG1.norm("l2")], [0.])

    if collective_rank == 0:
        print('Yes this is a consistent parallel parititioning')


