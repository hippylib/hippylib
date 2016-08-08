# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import dolfin as dl
import numpy as np
import os
    
abspath = os.path.dirname( os.path.abspath(__file__) )
sdir = os.path.join(abspath,"AssemblePointwiseObservation")
header_file = open(os.path.join(sdir,"AssemblePointwiseObservation.h"), "r")
code = header_file.read()
header_file.close()
#check the dolfin version to decide which cpp to include
if dl.dolfin_version()[2] == "4":
    cpp_sources = ["AssemblePointwiseObservation_v14.cpp"]
elif dl.dolfin_version()[2] == "5":
    cpp_sources = ["AssemblePointwiseObservation_v15.cpp"]
elif dl.dolfin_version()[2] == "6":
    cpp_sources = ["AssemblePointwiseObservation_v16.cpp"]
else:
    raise Exception("Dolfin Version")

cpp_module = dl.compile_extension_module(
code=code, source_directory=sdir, sources=cpp_sources,
include_dirs=[".",  sdir])

def assemblePointwiseObservation(Vh, targets):
    """
    Assemble the pointwise observation matrix:
    Input
    - Vh: FEniCS finite element space
    - targets: observation points (numpy array)
     
    Note: This Function will not work in parallel!!!
    """
    #Ensure that PetscInitialize is called
    dummy = dl.assemble( dl.inner(dl.TrialFunction(Vh), dl.TestFunction(Vh))*dl.dx )
    #Call the cpp module to compute the pointwise observation matrix
    tmp = cpp_module.PointwiseObservation(Vh,targets.flatten())
    #return the matrix
    return tmp.GetMatrix()

def exportPointwiseObservation(points, data, fname, varname="observation"):
    """
    This function write a VTK PolyData file to visualize pointwise data.
    Inputs:
    - points:  locations of the points 
               (numpy array of size number of points by space dimension)
    - data:    pointwise values
               (dolfin vector of size number of points)
    - fname:   filename for the vtk file to export
    - varname: name of the variable for the vtk file
    """
    ndim = points.shape[1]
    npoints = points.shape[0]
    
    if ndim == 2:
        points3d = np.zeros((npoints,3), dtype = points.dtype)
        points3d[:,:-1] = points
        exportPointwiseObservation(points3d, data, fname, varname)
        return
    
    f = open(fname, 'w')
    f.write('<VTKFile version="0.1" byte_order="LittleEndian" type="PolyData">\n')
    f.write('<PolyData>\n')
    f.write('<Piece NumberOfPoints="{0:d}" NumberOfVerts="{1:d}">\n'.format(npoints,npoints))
    f.write('<Points>\n')
    f.write('<DataArray NumberOfComponents="{0:d}" format="ascii" type="Float32">\n'.format(ndim))
    f.write( np.array_str(points).replace("[", "").replace("]", "") )
    f.write('\n</DataArray>\n')
    f.write('</Points>\n')
    f.write('<Verts>\n')
    f.write('\n<DataArray type="Int32" Name="connectivity" format="ascii">\n')
    f.write(np.array_str( np.arange(0,npoints) ).replace("[", "").replace("]", "") )
    f.write('</DataArray>\n')
    f.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
    f.write(np.array_str( np.arange(1,npoints+1) ).replace("[", "").replace("]", "") )
    f.write('</DataArray>\n')
    f.write('</Verts>\n')
    f.write('<PointData Scalars="{}">\n'.format(varname))
    f.write('<DataArray format="ascii" type="Float32" Name="{}">\n'.format(varname))
    f.write(np.array_str( data.array() ).replace("[", "").replace("]", "") )
    f.write('\n</DataArray>\n</PointData>\n<CellData>\n</CellData>\n</Piece>\n</PolyData>\n</VTKFile>')

    
    

#    from petsc4py import PETSc
#    import numpy as np
#
#    def assemblePointwiseObservation(Vh, targets):
#        """
#        Assemble the pointwise observation matrix:
#        Input
#        - Vh: FEniCS finite element space
#        - targets: observation points (numpy array)
#         
#        Note: This Function will not work in parallel!!!
#        """
#        ntargets, dim = targets.shape
#        mesh = Vh.mesh()
#        coords = mesh.coordinates()
#        cells = mesh.cells()
#        dolfin_element = Vh.dolfin_element()
#        dofmap = Vh.dofmap()
#        bbt = mesh.bounding_box_tree()
#        sdim = dolfin_element.space_dimension()
#        v = np.zeros(sdim)
#        
#        A = PETSc.Mat()
#        A.create(mesh.mpi_comm())
#        A.setSizes([ntargets, Vh.dim()])
#        A.setType("aij")
#        A.setPreallocationNNZ(sdim*ntargets)
#        
#        # In Parallel we will need to fix the rowLGmap so that only the points in the
#        # local mesh are kept.    
#        rowLGmap = PETSc.LGMap().create(range(0,ntargets), comm = mesh.mpi_comm() )
#        colLGmap = PETSc.LGMap().create(dofmap.dofs(), comm = mesh.mpi_comm() )
#        A.setLGMap(rowLGmap, colLGmap)
#        
#        for k in range(ntargets):
#            t = targets[k,:]
#            p = dl.Point(t)
#            cell_id = bbt.compute_first_entity_collision(p)
#            tvert = coords[cells[cell_id,:],:]
#            dolfin_element.evaluate_basis_all(v,t,tvert, cell_id)
#            cols = dofmap.cell_dofs(cell_id)
#            for j in range(sdim):
#                A[k, cols[j]] = v[j]
#                
#        A.assemblyBegin()
#        A.assemblyEnd()
#         
#        return dl.Matrix( dl.PETScMatrix(A) )