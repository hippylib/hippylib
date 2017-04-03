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

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/Matrix.h>

namespace dolfin
{

class PointwiseObservation
{
public:
	PointwiseObservation(const FunctionSpace & Vh, const Array<double> & targets);
	std::shared_ptr<Matrix> GetMatrix();
	~PointwiseObservation();

private:
	PetscInt computeLGtargets(MPI_Comm comm, std::shared_ptr<BoundingBoxTree> bbt,
					 const std::size_t gdim,
					 const Array<double> & targets,
					 std::vector<Point> & points,
			         std::vector<PetscInt> & LG);
	Mat mat;
	std::vector<int> old_new;
};

}
