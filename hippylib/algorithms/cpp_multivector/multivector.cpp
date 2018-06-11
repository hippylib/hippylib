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

#include "multivector.h"
#include <dolfin/la/PETScVector.h>

#include <cassert>

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


void MultiVector::dot(const GenericVector & v, Array<double> & m)
{
	double* im = m.data();
	for(auto&& vj : mv)
		*(im++) = vj->inner(v);
}

void MultiVector::dot(const MultiVector & other, Array<double> & m)
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

void MultiVector::dot_self(Array<double> & m)
{
	int s = mv.size();
	for(int i = 0; i < s; ++i)
	{
		m[i + s*i] = mv[i]->inner(*(mv[i]));
		for(int j = 0; j < i; ++j)
			m[i + s*j] = m[j + s*i] = mv[i]->inner(*(mv[j]));

	}
}

void MultiVector::reduce(GenericVector & v, const Array<double> & alpha)
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

void MultiVector::axpy(const Array<double> & a, const MultiVector & y)
{
	int n = nvec();
	assert(a.size() == n);
	assert(y.nvec() == n);

	for(int i = 0; i < n; ++i)
		mv[i]->axpy(a[i], *(y.mv[i]) );
}

void MultiVector::scale(int k, double a)
{
	mv[k]->operator*=(a);
}

void MultiVector::scale(const Array<double> & a)
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

void MultiVector::norm_all(const std::string norm_type, Array<double> & norms)
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
