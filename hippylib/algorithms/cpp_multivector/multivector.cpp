/* Copyright (c) 2016-2018, The University of Texas at Austin
 * & University of California--Merced.
 * Copyright (c) 2019-2020, The University of Texas at Austin,
 * University of California--Merced, Washington University in St. Louis.
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

#include "multivector.hpp"
#include <dolfin/la/PETScVector.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cassert>

namespace py = pybind11;

namespace hippylib
{

MultiVector::MultiVector()
{
}

MultiVector::MultiVector(const dolfin::GenericVector & v, std::size_t nvec):
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
	std::size_t n = mv.size();
	for(std::size_t i = 0; i < n; ++i)
		mv[i] = orig.mv[i]->copy();
}

void MultiVector::setSizeFromVector(const dolfin::GenericVector & v, std::size_t nvec)
{
	mv.resize(nvec);
	for(auto&& vj : mv)
	{
		vj = v.copy();
		vj->zero();
	}
}



std::shared_ptr<const dolfin::GenericVector> MultiVector::operator[](std::size_t i) const
{
	return mv[i];
}

std::shared_ptr<dolfin::GenericVector> MultiVector::operator[](std::size_t i)
{
	return mv[i];
}


void MultiVector::dot(const dolfin::GenericVector & v, dolfin::Array<double> & m) const
{
	double* im = m.data();
	for(auto&& vj : mv)
		*(im++) = vj->inner(v);
}

void MultiVector::dot(const MultiVector & other, dolfin::Array<double> & m) const
{
	if(other.mv.begin() == mv.begin())
		dot_self(m);
	else
	{
		double* data = m.data();
		for(auto&& vi : mv)
			for(auto&& vj : other.mv)
				*(data++) = vi->inner(*vj);
	}
}

void MultiVector::dot_self(dolfin::Array<double> & m) const
{
	std::size_t s = mv.size();
	for(std::size_t i = 0; i < s; ++i)
	{
		m[i + s*i] = mv[i]->inner(*(mv[i]));
		for(std::size_t j = 0; j < i; ++j)
			m[i + s*j] = m[j + s*i] = mv[i]->inner(*(mv[j]));

	}
}

void MultiVector::reduce(dolfin::GenericVector & v, const dolfin::Array<double> & alpha) const
{
	const double * data = alpha.data();
	for(auto&& vi : mv)
		v.axpy(*(data++), *vi);
}

void MultiVector::axpy(double a, const dolfin::GenericVector & y)
{
	for(auto&& vi : mv)
		vi->axpy(a, y);
}

void MultiVector::axpy(const dolfin::Array<double> & a, const MultiVector & y)
{
	std::size_t n = nvec();
	assert(a.size() == n);
	assert(y.nvec() == n);

	for(std::size_t i = 0; i < n; ++i)
		mv[i]->axpy(a[i], *(y.mv[i]) );
}

void MultiVector::scale(std::size_t k, double a)
{
	mv[k]->operator*=(a);
}

void MultiVector::scale(const dolfin::Array<double> & a)
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

void MultiVector::norm_all(const std::string norm_type, dolfin::Array<double> & norms) const
{
	double * data = norms.data();
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

}

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<hippylib::MultiVector>(m, "MultiVector")
    	.def(py::init<>())
		.def(py::init<const dolfin::GenericVector &, std::size_t>())
		.def(py::init<const hippylib::MultiVector &>())
		.def("nvec", &hippylib::MultiVector::nvec,
			 "Number of vectors in the multivector")
		.def("__len__", &hippylib::MultiVector::nvec,
			 "The length of a multivector is the number of vector it contains")
		.def("__getitem__", (std::shared_ptr<const dolfin::GenericVector> (hippylib::MultiVector::*)(std::size_t) const) &hippylib::MultiVector::operator[] )
		.def("__setitem__", (std::shared_ptr<dolfin::GenericVector> (hippylib::MultiVector::*)(std::size_t)) &hippylib::MultiVector::operator[] )
		.def("setSizeFromVector", &hippylib::MultiVector::setSizeFromVector,
			 "Initialize a multivector by providing a vector v as template and the number of vectors nvec",
			 py::arg("v"), py::arg("nvec"))
		.def("dot", [](const hippylib::MultiVector & self, const dolfin::GenericVector & v)
				{
    				std::size_t size = self.nvec();
    				py::array_t<double> ma(size);
    				dolfin::Array<double> m_dolfin(size, ma.mutable_data());
    				self.dot(v, m_dolfin);
    				return ma;
				},
				"Perform the inner product with a vector v",
				py::arg("v")
    	     )
		.def("dot", [](const hippylib::MultiVector& self, const hippylib::MultiVector & other)
				{
    				std::size_t size1 = self.nvec();
    				std::size_t size2 = other.nvec();
    				py::array_t<double> ma(size1*size2);
    				dolfin::Array<double> m_dolfin(size1*size2, ma.mutable_data());
    				self.dot(other, m_dolfin);
    				return ma;
				},
    			"Perform the inner product with a another multivector",
				py::arg("other")
			)
        .def("reduce", [](const hippylib::MultiVector& self, dolfin::GenericVector & v, py::array_t<double> & alpha)
        		{
    				dolfin::Array<double> alpha_dolfin(self.nvec(), alpha.mutable_data());
    				self.reduce(v, alpha_dolfin);
        		},
    		"Computes v += sum_i alpha[i]*self[i]",
			py::arg("v"), py::arg("alpha")
        )
		.def("axpy", (void (hippylib::MultiVector::*)(double, const dolfin::GenericVector &)) &hippylib::MultiVector::axpy,
			 "Assign self[k] += a*y for k in range(self.nvec())",
			 py::arg("a"), py::arg("y"))
		.def("axpy", [](hippylib::MultiVector& self, py::array_t<double> & a, const hippylib::MultiVector& y)
				{
    				dolfin::Array<double> a_dolfin(self.nvec(), a.mutable_data());
    				self.axpy(a_dolfin, y);
				},
				"Assign self[k] += a[k]*y[k] for k in range(self.nvec())",
				py::arg("a"), py::arg("y")
				)
		.def("scale", (void (hippylib::MultiVector::*)(std::size_t, double)) &hippylib::MultiVector::scale,
			 "Assign self[k] *= a",
			 py::arg("k"), py::arg("a"))
		.def("scale", [](hippylib::MultiVector& self, py::array_t<double> & a)
				{
					dolfin::Array<double> a_dolfin(self.nvec(), a.mutable_data());
					self.scale(a_dolfin);
				},
				"Assign self[k] *=a[k] for k in range(self.nvec()",
				py::arg("a")
				)
		.def("zero",  &hippylib::MultiVector::zero,
			 "Zero out all entries of the multivector"
			)
		.def("norm",[](const hippylib::MultiVector& self, const std::string norm_type)
				{
					std::size_t size = self.nvec();
					py::array_t<double> ma(size);
					dolfin::Array<double> m_dolfin(size, ma.mutable_data());
					self.norm_all(norm_type, m_dolfin);
					return ma;
				},
			 "Compute the norm of each vector in the multivector separately",
			 py::arg("norm_type")
			)
		.def("swap", &hippylib::MultiVector::swap,
			 "Swap this with other");
}
