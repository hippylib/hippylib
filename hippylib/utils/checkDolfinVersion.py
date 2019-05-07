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

import dolfin as dl

def dlversion():
    version_str = dl.__version__
    vv = [int(ss) for ss in version_str.split(".")]
    
    return (vv[0], vv[1], vv[2])

supported_versions = [(2018,1,0), (2019,1,0)]

def checkdlversion():
    """
    Check if :code:`FEniCS` version is supported. Currently :code:`hIPPYlib` requires
    :code:`FEniCS` version :code:`1.6.0` and newer.
    """
    if dlversion() not in supported_versions:
        print( "The version of FEniCS (FEniCS {0}.{1}.{2}) you are using is not supported.".format(*dlversion()) )
        exit()
        
if  dlversion() >= (2017,2,0):
    import matplotlib.pyplot as plt
    dl.interactive = plt.show
    
    import warnings
    from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
    warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)