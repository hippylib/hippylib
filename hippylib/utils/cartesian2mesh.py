'''
Created on Jul 31, 2019

@author: uvilla
'''
import dolfin as dl
import os
    
abspath = os.path.dirname( os.path.abspath(__file__) )
source_directory = os.path.join(abspath,"cpp_cartesian2mesh")
cpp_code = ""
for fname in ["numpy2MeshFunction.cpp",  "numpy2Expression.cpp", "pybind_module.cpp"]:
    with open(os.path.join(source_directory, fname), "r") as cpp_file:
        cpp_code    += cpp_file.read()

include_dirs = [".", source_directory]
cpp_module = dl.compile_cpp_code(cpp_code, include_dirs=include_dirs)

numpy2MeshFunction = cpp_module.numpy2MeshFunction
NumpyScalarExpression3D = cpp_module.NumpyScalarExpression3D
NumpyScalarExpression2D = cpp_module.NumpyScalarExpression2D
NumpyScalarExpression1D = cpp_module.NumpyScalarExpression1D