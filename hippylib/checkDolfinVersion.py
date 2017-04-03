'''
Created on Nov 22, 2016

@author: uvilla
'''
import dolfin as dl

def dlversion():
    return (dl.DOLFIN_VERSION_MAJOR, dl.DOLFIN_VERSION_MINOR, dl.DOLFIN_VERSION_MICRO)

supported_versions = [(1,6,0), (2016,1,0), (2016,2,0)]

def checkdlversion():
    if dlversion() not in supported_versions:
        print "The version of FEniCS (FEniCS {0}.{1}.{2}) you are using is not supported.".format(*dlversion())
        exit()