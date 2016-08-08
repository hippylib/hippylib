/* Copyright (c) 2016, The University of Texas at Austin & University of
 * California, Merced.
 *
 * All Rights reserved.
 * See file COPYRIGHT for details.
 *
 * This file is part of the hIPPYlib library. For more information and source
 * code availability see https://hippylib.github.io.
 *
 * hIPPYlib is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 3.0 dated June 2007.
*/

#include "AssemblePointwiseObservation.h"
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <vector>
#include <set>
#include <cassert>

using namespace dolfin;


PointwiseObservation::PointwiseObservation(const FunctionSpace & Vh, const Array<double> & targets)
{
	 const Mesh& mesh = *( Vh.mesh() );
	 MPI_Comm comm = mesh.mpi_comm();
	 int nprocs;
	 MPI_Comm_size(comm, &nprocs);
	 if(nprocs != 1)
	 {
		 std::cout << "PointwiseObservation::PointwiseObservation is only serial.\n";
		 exit(1);
	 }
	 std::size_t gdim = mesh.geometry().dim();

	 std::size_t ntargets = targets.size() / gdim;
	 assert(ntargets*gdim == targets.size() );

	 std::vector<Point> points(0);
	 points.reserve(ntargets);

	 for(int i = 0; i < ntargets; ++i)
		 points.push_back( Point(gdim, targets.data()+i*gdim) );

	 std::shared_ptr<BoundingBoxTree> bbt = mesh.bounding_box_tree();

	 std::shared_ptr<const FiniteElement> element( Vh.element() );
	 //Check that value_rank is either 0 (scalar FE or 1 vector FE)
	 assert(element->value_rank() == 0 || element->value_rank() == 1);
	 int value_dim = element->value_dimension(0);

	 std::shared_ptr<const GenericDofMap> dofmap = Vh.dofmap();
	 PetscInt global_dof_dimension = dofmap->global_dimension();
	 PetscInt local_dof_dimension = dofmap->local_dimension("owned");
	 std::vector<dolfin::la_index> LGdofs = dofmap->dofs();

	 PetscInt global_ntargets = points.size();
	 std::vector<PetscInt> LGtargets(0);
	 LGtargets.reserve(global_ntargets);
	 PetscInt lt = 0;
	 for(int gt = 0; gt < global_ntargets; ++gt)
		 if( bbt->collides_entity(points[gt]) )
		 {
			 LGtargets.push_back(lt);
			 ++lt;
		 }
	 PetscInt local_ntargets  = lt;

	 PetscInt global_nrows = global_ntargets*value_dim;
	 PetscInt local_nrows = local_ntargets*value_dim;
	 std::vector<PetscInt> LGrows(local_nrows);
	 int counter = 0;

	 for(int lt = 0; lt < local_ntargets; ++lt)
		 for(int ival = 0; ival < value_dim; ++ival, ++ counter)
			 LGrows[counter] = lt*value_dim + ival;

	 MatCreate(comm,&mat);
	 MatSetSizes(mat,local_nrows,local_dof_dimension,global_nrows,global_dof_dimension);
	 MatSetType(mat,MATAIJ);
	 MatSetUp(mat);
	 ISLocalToGlobalMapping rmapping, cmapping;
	 PetscCopyMode mode = PETSC_COPY_VALUES;
#if PETSC_VERSION_LT(3,5,0)
	 ISLocalToGlobalMappingCreate(comm, LGrows.size(), &LGrows[0], mode, &rmapping);
	 ISLocalToGlobalMappingCreate(comm, LGdofs.size(),&LGdofs[0],mode,&cmapping);
#else
	 PetscInt bs = 1;
	 ISLocalToGlobalMappingCreate(comm, bs, LGrows.size(), &LGrows[0], mode, &rmapping);
	 ISLocalToGlobalMappingCreate(comm, bs, LGdofs.size(),&LGdofs[0],mode,&cmapping);
#endif
	 MatSetLocalToGlobalMapping(mat,rmapping,cmapping);

	 //Space dimension is the local number of Dofs
	 std::size_t sdim = element->space_dimension();
	 std::vector<double> basis_matrix(sdim*value_dim);
	 std::vector<double> basis_matrix_row_major(sdim*value_dim);
	 std::vector<PetscInt> cols(sdim);

	 for(int lt = 0; lt < local_ntargets; ++lt)
	 {
		 int gt = LGtargets[lt];
		 int cell_id = bbt->compute_first_entity_collision(points[gt]);

		 Cell cell(mesh, cell_id);
		 std::vector<double> coords;
		 cell.get_vertex_coordinates(coords);
		 element->evaluate_basis_all(&basis_matrix[0], points[gt].coordinates(), &coords[0], cell.orientation());
		 const std::vector<dolfin::la_index>& cell_dofs = dofmap->cell_dofs(cell_id);
		 std::copy(cell_dofs.begin(), cell_dofs.end(), cols.begin());
		 for(int i = 0; i < sdim; ++i)
			 for(int j = 0; j < value_dim; ++j)
				 basis_matrix_row_major[i+j*sdim] = basis_matrix[value_dim*i+j];

		 MatSetValues(mat,value_dim,&LGrows[value_dim*gt],sdim, &cols[0],&basis_matrix_row_major[0],INSERT_VALUES);
	 }

	 MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
	 MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
}

Matrix PointwiseObservation::GetMatrix()
{
	return Matrix( PETScMatrix(mat) );
}
