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

#include <pybind11/pybind11.h>
#include <dolfin/la/PETScVector.h>
#include "PRNG.h"

namespace py = pybind11;

namespace dolfin{
Random::Random(int seed):
		eng(seed),
		d_normal(0.,1.),
		d_uniform(0., 1.)
{

}

void Random::split(int _rank, int _nproc, int _block_size)
{
	eng.split(_rank, _nproc, _block_size);
}

double Random::uniform(double a, double b)
{
	double r = d_uniform( eng );
	return a + (b-a)*r;
}

double Random::normal(double mu, double sigma)
{
	double z = d_normal( eng );
	return mu + sigma*z;
}

double Random::rademacher()
{
	bool val = d_bernoulli( eng );
	if(val)
		return 1.;
	else
		return -1.;
}

void Random::uniform(GenericVector & v, double a, double b)
{
	PETScVector* vec = &as_type<PETScVector>(v);
	Vec vv = vec->vec();

	PetscInt local_size;
	VecGetLocalSize(vv, &local_size);

	PetscScalar *data = NULL;
	VecGetArray(vv, &data);

	for(PetscInt i = 0; i < local_size; ++i)
		data[i] = a + (b-a)*d_uniform( eng );

	VecRestoreArray(vv, &data);
}

void Random::normal(GenericVector & v, double sigma, bool zero_out)
{
	if(zero_out)
		v.zero();

	PETScVector* vec = &as_type<PETScVector>(v);
	Vec vv = vec->vec();

	PetscInt local_size;
	VecGetLocalSize(vv, &local_size);

	PetscScalar *data = NULL;
	VecGetArray(vv, &data);

	for(PetscInt i = 0; i < local_size; ++i)
		data[i] += sigma*d_normal( eng );

	VecRestoreArray(vv, &data);
}

void Random::rademacher(GenericVector & v)
{
	PETScVector* vec = &as_type<PETScVector>(v);
	Vec vv = vec->vec();

	PetscInt local_size;
	VecGetLocalSize(vv, &local_size);

	PetscScalar *data = NULL;
	VecGetArray(vv, &data);

	for(PetscInt i = 0; i < local_size; ++i)
	{
		auto val = d_bernoulli( eng );
		if(val)
			data[i] = 1.;
		else
			data[i] = -1.;
	}

	VecRestoreArray(vv, &data);
}
}

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<dolfin::Random>(m, "Random")
    	.def(py::init<int>())
        .def("split", &dolfin::Random::split,
        	 "Split the random number generator in independent streams",
			 py::arg("rank"), py::arg("nproc"), py::arg("block_size"))
		.def("uniform", (double (dolfin::Random::*)(double, double)) &dolfin::Random::uniform,
			"Generate a sample from U(a,b)",
			py::arg("a")=0., py::arg("b")=1.)
		.def("uniform", (void (dolfin::Random::*)(dolfin::GenericVector &, double, double)) &dolfin::Random::uniform,
			"Generate a random vector from U(a,b)",
			py::arg("out"), py::arg("a")=0., py::arg("b")=1.)
		.def("normal", (double (dolfin::Random::*)(double, double)) &dolfin::Random::normal,
			"Generate a sample from N(mu, sigma2)",
			py::arg("mu")=0., py::arg("sigma")=1.)
		.def("normal", (void (dolfin::Random::*)(dolfin::GenericVector &, double, bool)) &dolfin::Random::normal,
			"Generate a random vector from N(0, sigma2)",
			py::arg("out"), py::arg("sigma")=1., py::arg("zero_out")=true)
		.def("rademacher", (double (dolfin::Random::*)(void)) &dolfin::Random::rademacher,
			"Generate a sample from Rademacher distribution")
		.def("rademacher", (void (dolfin::Random::*)(dolfin::GenericVector &)) &dolfin::Random::rademacher,
			 "Generate a random vector from Rademacher distribution",
			 py::arg("out"));
}
