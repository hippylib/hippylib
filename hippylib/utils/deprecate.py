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
import warnings
import os

from .warnings import hIPPYlibDeprecationWarning

warnings.filterwarnings(os.environ.get("hIPPYlibDeprecationWarning", "once"), category=hIPPYlibDeprecationWarning)

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
    _name = name
    def deprecated_function(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            name = f.__name__ if _name is None else _name
            warnings.warn("WARNING: {0} DEPRECATED since v{1}. {2}".format(name, version, msg),
                      category=hIPPYlibDeprecationWarning,
                      stacklevel=2)
            return f(*args, **kwargs)
        return wrapped
    return deprecated_function
