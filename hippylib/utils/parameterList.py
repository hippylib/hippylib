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

class ParameterList(object):
    """
    A small abstract class for storing parameters and their description.
    This class will raise an exception if the key one tries to access is not present.
    """
    def __init__(self, data):
        """
        data is a dictionary where each value is the pair (value, description)
        """
        self.data = data
        
    def __getitem__(self,key):
        if self.data.__contains__(key):
            return self.data[key][0]
        else:
            raise ValueError(key)
        
    def __setitem__(self,key, value):
        if self.data.__contains__(key):
            self.data[key][0] = value
        else:
            raise ValueError(key)
        
    def showMe(self, indent=""):
        for k in sorted(self.data.keys()):
            print( indent, "---")
            if type(self.data[k][0]) == ParameterList:
                print( indent, k, "(ParameterList):", self.data[k][1] )
                self.data[k][0].showMe(indent+"    ")
            else:
                print( indent, k, "({0}):".format(self.data[k][0]),  self.data[k][1] )
        
        print( indent, "---")