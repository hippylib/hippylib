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

from functools import wraps
import sys
import warnings

#if -W warning flag not set, by default warn only once per python session.
if not sys.warnoptions:
    warnings.simplefilter("once")

def deprecated(name=None, version=None, msg=""):
    """
    A decorator to designate functions as deprecated. A warning is given
    when the function is called. By default, warnings are only given once
    per python session.
    
    Keyword args:
      name (str): name of the function or function call that is deprecated (optional)
      version (str): the version the function was deprecated (required)
      msg (str): message to the user, typically providing alternative function calls 
                 and/or notice of version for removal of deprecated function (optional)
    """
    if type(version) is not str:
        raise ValueError ("'version' must be a string corresponding to the version number.")
    if name is not None:
        if type(name) is not str:
            raise ValueError ("'name' must be a string corresponding to the function call that is deprecated.")
    else:
        name = f.__name__    
    def deprecated_function(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            warnings.warn("WARNING: %s DEPRECATED since v%s. %s" % (name, version, msg),
                      category=DeprecationWarning,
                      stacklevel=2)
            return f(*args, **kwargs)
        return wrapped
    return deprecated_function
