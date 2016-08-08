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

#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/PETScMatrix.h>

namespace dolfin
{

class cpp_linalg
{
public:
	//out = A*B
	Matrix MatMatMult(const GenericMatrix & A, const GenericMatrix & B);
	//out = Pt*A*P
	Matrix MatPtAP(const GenericMatrix & A, const GenericMatrix & P);
	//out = At*B
	Matrix MatAtB(const GenericMatrix & A, const GenericMatrix & B);
	//out = At
	Matrix Transpose(const GenericMatrix & A);
};

}
