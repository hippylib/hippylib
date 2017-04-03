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

#include <dolfin.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <vector>
#include <set>
#include <cassert>

#include "AssemblePointwiseObservation.h"

using namespace dolfin;

PetscInt PointwiseObservation::computeLGtargets(MPI_Comm comm,
		                               std::shared_ptr<BoundingBoxTree> bbt,
									   const std::size_t gdim,
									   const Array<double> & targets,
									   std::vector<Point> & points,
		                               std::vector<PetscInt> & LG)
{
	 int nprocs, rank;
	 MPI_Comm_size(comm, &nprocs);
	 MPI_Comm_rank(comm, &rank);

	 const std::size_t nTargets = targets.size()/gdim;

	 points.reserve(nTargets);
	 LG.reserve(nTargets);
	 std::vector<int> tmp(nTargets);
	 for(int i = 0; i < nTargets; ++i)
	 {
		 Point p(gdim, targets.data()+i*gdim);
		 if(bbt->collides_entity(p))
	 	 	 tmp[i] = rank;
	 	 else
	 		 tmp[i] = nprocs;
	 }

	 std::vector<int> owner(nTargets);
	 MPI_Allreduce(tmp.data(), owner.data(), nTargets, MPI_INT, MPI_MIN, comm);

	 for(int i = 0; i < nTargets; ++i)
		 if( owner[i] == rank )
		 {
			 LG.push_back(i);
			 points.push_back( Point(gdim, targets.data()+i*gdim) );
		 }

	 std::vector<PetscInt> proc_offset(nprocs+1);
	 std::fill(proc_offset.begin(), proc_offset.end(), 0);
	 for(int i = 0; i < nTargets; ++i)
		 ++proc_offset[owner[i]+1];

	 std::partial_sum(proc_offset.begin(), proc_offset.end(), proc_offset.begin());

	 old_new.resize(nTargets);
	 for(int i = 0; i < nTargets; ++i)
	 {
		 old_new[i] = proc_offset[owner[i]];
		 ++proc_offset[owner[i]];
	 }

	 for(int jj = 0; jj < LG.size(); ++jj)
		 LG[jj] = old_new[LG[jj]];

	 return proc_offset[nprocs];
}

PointwiseObservation::PointwiseObservation(const FunctionSpace & Vh, const Array<double> & targets)
{
	 const Mesh& mesh = *( Vh.mesh() );
	 const int num_cells = mesh.num_cells();
	 MPI_Comm comm = mesh.mpi_comm();
	 int nprocs, rank;
	 MPI_Comm_size(comm, &nprocs);
	 MPI_Comm_rank(comm, &rank);

	 const std::size_t gdim(mesh.geometry().dim());

	 const std::size_t ntargets(targets.size() / gdim);
	 assert(ntargets*gdim == targets.size() );

	 std::shared_ptr<BoundingBoxTree> bbt = mesh.bounding_box_tree();

	 std::vector<Point> points(0);
	 std::vector<PetscInt> LGtargets(0);

	 PetscInt global_ntargets = computeLGtargets(comm, bbt, gdim, targets, points, LGtargets);
	 PetscInt local_ntargets  = points.size();

	 std::shared_ptr<const FiniteElement> element( Vh.element() );
	 //Check that value_rank is either 0 (scalar FE or 1 vector FE)
	 assert(element->value_rank() == 0 || element->value_rank() == 1);
	 int value_dim = element->value_dimension(0);

	 std::shared_ptr<const GenericDofMap> dofmap = Vh.dofmap();
	 PetscInt global_dof_dimension = dofmap->global_dimension();
#if DOLFIN_VERSION_MAJOR >= 2016
	 PetscInt local_dof_dimension = dofmap->index_map()->size(IndexMap::MapSize::OWNED);
#else
	 PetscInt local_dof_dimension = dofmap->local_dimension("owned");
#endif
	 std::vector<dolfin::la_index> LGdofs = dofmap->dofs();

	 PetscInt global_nrows = global_ntargets*value_dim;
	 PetscInt local_nrows = local_ntargets*value_dim;
	 std::vector<PetscInt> LGrows(local_nrows);
	 int counter = 0;
	 for(int lt = 0; lt < local_ntargets; ++lt)
		 for(int ival = 0; ival < value_dim; ++ival, ++ counter)
			 LGrows[counter] = LGtargets[lt]*value_dim + ival;

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
		 int cell_id = bbt->compute_first_entity_collision(points[lt]);

		 if(cell_id < 0 or cell_id > num_cells)
		 {
			 std::cout << "Pid" << rank << ": Something went wrong. cell_id is " << cell_id << std::endl;
			 MPI_Abort(comm, 1);
		 }

		 Cell cell(mesh, cell_id);
		 std::vector<double> coords;
		 cell.get_vertex_coordinates(coords);
		 element->evaluate_basis_all(&basis_matrix[0], points[lt].coordinates(), &coords[0], cell.orientation());
		 ArrayView<const dolfin::la_index> cell_dofs = dofmap->cell_dofs(cell_id);
		 auto it_col = cols.begin();
		 for(auto it = cell_dofs.begin(); it != cell_dofs.end(); ++it, ++it_col)
			 *it_col = dofmap->local_to_global_index(*it);
		 for(int i = 0; i < sdim; ++i)
			 for(int j = 0; j < value_dim; ++j)
				 basis_matrix_row_major[i+j*sdim] = basis_matrix[value_dim*i+j];

		 PetscErrorCode ierr = MatSetValues(mat,value_dim,&LGrows[value_dim*lt],sdim, &cols[0],&basis_matrix_row_major[0],INSERT_VALUES);
		 if (ierr != 0)
			 std::cout << "Rank "<< rank << "lt = " << lt << std::endl;
	 }

	 MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
	 MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
}

PointwiseObservation::~PointwiseObservation()
{
	MatDestroy(&mat);
}

std::shared_ptr<Matrix> PointwiseObservation::GetMatrix()
{
	return std::shared_ptr<Matrix>( new Matrix( PETScMatrix(mat) ) );
}
