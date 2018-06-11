/* Copyright (c) 2016-2018, The University of Texas at Austin
 * & University of California, Merced.
 *
 * All Rights reserved.
 * See file COPYRIGHT for details.
 *
 * This file is part of the hIPPYlib library. For more information and source
 * code availability see https://hippylib.github.io.
 *
 * hIPPYlib is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.0 dated June 1991.
*/

#include "linalg.h"
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/GenericLinearSolver.h>
#include <dolfin/common/Timer.h>

#include <cassert>

namespace dolfin
{

std::shared_ptr<Matrix> cpp_linalg::MatMatMult(const GenericMatrix & A, const GenericMatrix & B)
{
    const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
    const PETScMatrix* Bp = &as_type<const PETScMatrix>(B);
    Mat CC;
    PetscErrorCode ierr = ::MatMatMult(Ap->mat(), Bp->mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CC);

    ISLocalToGlobalMapping rmappingA;
    ISLocalToGlobalMapping cmappingB;
    MatGetLocalToGlobalMapping(Ap->mat(),&rmappingA,NULL);
    MatGetLocalToGlobalMapping(Bp->mat(),NULL, &cmappingB);

    MatSetLocalToGlobalMapping(CC, rmappingA, cmappingB);

    PETScMatrix CCC = PETScMatrix(CC);
    MatDestroy(&CC);
    return std::shared_ptr<Matrix>( new Matrix(CCC) );
}

std::shared_ptr<Matrix> cpp_linalg::MatPtAP(const GenericMatrix & A, const GenericMatrix & P)
{
	const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
	const PETScMatrix* Pp = &as_type<const PETScMatrix>(P);
    Mat CC;
    PetscErrorCode ierr = ::MatPtAP(Ap->mat(),Pp->mat(),MAT_INITIAL_MATRIX, 1.0,&CC);

    //Manually set the LocalToGlobalMapping
    ISLocalToGlobalMapping mapping;
    MatGetLocalToGlobalMapping(Pp->mat(),NULL, &mapping);
    MatSetLocalToGlobalMapping(CC, mapping, mapping);


    PETScMatrix CCC = PETScMatrix(CC);
    MatDestroy(&CC);
    return std::shared_ptr<Matrix>( new Matrix(CCC) );
}

std::shared_ptr<Matrix> cpp_linalg::MatAtB(const GenericMatrix & A, const GenericMatrix & B)
{
    const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
    const PETScMatrix* Bp = &as_type<const PETScMatrix>(B);
    Mat CC;
    PetscErrorCode ierr = MatTransposeMatMult(Ap->mat(), Bp->mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CC);

    ISLocalToGlobalMapping cmappingA;
    ISLocalToGlobalMapping cmappingB;
    MatGetLocalToGlobalMapping(Ap->mat(),NULL, &cmappingA);
    MatGetLocalToGlobalMapping(Bp->mat(),NULL, &cmappingB);

    MatSetLocalToGlobalMapping(CC, cmappingA, cmappingB);

    PETScMatrix CCC = PETScMatrix(CC);
    MatDestroy(&CC);
    return std::shared_ptr<Matrix>(new Matrix(CCC) );
}

std::shared_ptr<Matrix> cpp_linalg::Transpose(const GenericMatrix & A)
{
	const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
	Mat At;
	MatTranspose(Ap->mat(), MAT_INITIAL_MATRIX, &At);

	ISLocalToGlobalMapping rmappingA;
	ISLocalToGlobalMapping cmappingA;
	MatGetLocalToGlobalMapping(Ap->mat(),&rmappingA, &cmappingA);
	MatSetLocalToGlobalMapping(At, cmappingA, rmappingA);

	std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(At)));
	MatDestroy(&At);
	return out;
}

void cpp_linalg::SetToOwnedGid(GenericVector & v, std::size_t gid, double val)
{
	assert(v.owns_index(gid));
	la_index index = static_cast<la_index>(gid);
	v.set(&val, 1, &index);
}

double cpp_linalg::GetFromOwnedGid(const GenericVector & v, std::size_t gid)
{
	assert(v.owns_index(gid));
	double val;
	la_index index = static_cast<la_index>(gid);
	v.get(&val, 1, &index);

	return val;
}

}
