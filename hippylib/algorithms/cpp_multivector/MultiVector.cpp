/*cppimport
<%                                                                               
from dolfin.jit.jit import dolfin_pc
flags = ["-D{}".format(i[0]) for i in dolfin_pc["define_macros"]]
cfg['libraries'] = dolfin_pc['libraries']                        
cfg['include_dirs'] = dolfin_pc['include_dirs']
cfg['library_dirs'] = dolfin_pc['library_dirs']
cfg['compiler_args'] += flags
setup_pybind11(cfg)          
%>                 
/*cppimport

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

#include "MultiVector.h"
#include <dolfin/la/PETScVector.h>

#include <cassert>

#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfin
{

MultiVector::MultiVector()
{
}

MultiVector::MultiVector(const GenericVector & v, int nvec):
		mv(nvec)
{
	for(auto&& vj : mv)
	{
		vj = v.copy();
		vj->zero();
	}
}

MultiVector::MultiVector(const MultiVector & orig):
		mv(orig.mv.size())
{
	int n = mv.size();
	for(int i = 0; i < n; ++i)
		mv[i] = orig.mv[i]->copy();
}

void MultiVector::setSizeFromVector(const GenericVector & v, int nvec)
{
	mv.resize(nvec);
	for(auto&& vj : mv)
	{
		vj = v.copy();
		vj->zero();
	}
}



std::shared_ptr<const GenericVector> MultiVector::operator[](int i) const
{
	return mv[i];
}

std::shared_ptr<GenericVector> MultiVector::operator[](int i)
{
	return mv[i];
}


void MultiVector::dot(const GenericVector & v, py::array_t<double> m)
{
	double* im = m.mutable_data();
	for(auto&& vj : mv)
		*(im++) = vj->inner(v);
}

void MultiVector::dot(const MultiVector & other, py::array_t<double> m)
{
	if(other.mv.begin() == mv.begin())
		dot_self(m);
	else
	{
		double* data = m.mutable_data();
		for(auto&& vi : mv)
			for(auto&& vj : other.mv)
				*(data++) = vi->inner(*vj);
	}
}

void MultiVector::dot_self(py::array_t<double> m)
{
	int s = mv.size();
	double* data = m.mutable_data();
	for(int i = 0; i < s; ++i)
	{
		data[i + s*i] = mv[i]->inner(*(mv[i]));
		for(int j = 0; j < i; ++j)
			data[i + s*j] = data[j + s*i] = mv[i]->inner(*(mv[j]));

	};
}

void MultiVector::reduce(GenericVector & v, const py::array_t<double> alpha)
{
	const double * data = alpha.data();
	for(auto&& vi : mv)
		v.axpy(*(data++), *vi);
}

void MultiVector::axpy(double a, const GenericVector & y)
{
	for(auto&& vi : mv)
		vi->axpy(a, y);
}

void MultiVector::axpy(const py::array_t<double> a, const MultiVector & y)
{
	int n = nvec();
	assert(a.size() == n);
	assert(y.nvec() == n);

	const double* data = a.data();

	for(int i = 0; i < n; ++i)
		mv[i]->axpy(data[i], *(y.mv[i]) );
}

void MultiVector::scale(int k, double a)
{
	mv[k]->operator*=(a);
}

void MultiVector::scale(const py::array_t<double> a)
{
	const double * data = a.data();
	for(auto && vj : mv)
		vj->operator*=(*(data++));
}

void MultiVector::zero()
{
	for(auto&& vi : mv)
		vi->zero();
}

void MultiVector::norm_all(const std::string norm_type, py::array_t<double> norms)
{
	double * data = norms.mutable_data();
	for(auto && vi : mv)
		*(data++) = vi->norm(norm_type);
}

void MultiVector::swap(MultiVector & other)
{
	mv.swap(other.mv);
}

MultiVector::~MultiVector()
{

}


PYBIND11_MODULE(MultiVector, m)
{
	py::class_<MultiVector, std::shared_ptr<MultiVector>>(m, "MultiVector")
        .def(py::init<>())
	.def(py::init<const GenericVector&, int>())
	.def(py::init<const MultiVector &>())
        .def("nvec", &MultiVector::nvec)
	.def("scale", (void (MultiVector::*)(int, double)) &MultiVector::scale)
	.def("dot", (void (MultiVector::*)(const GenericVector&, py::array_t<double>)) &MultiVector::dot)
	.def("dot", (void (MultiVector::*)(const MultiVector&, py::array_t<double>)) &MultiVector::dot)
	.def("__getitem__", [](MultiVector s, size_t i) {
            return s[i];
	})
	;

}
}
