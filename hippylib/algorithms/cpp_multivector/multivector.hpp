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


#include <dolfin/la/GenericVector.h>
#include <dolfin/common/Array.h>

namespace hippylib
{

class MultiVector
{
public:
	MultiVector();
	MultiVector(const dolfin::GenericVector & v, std::size_t nvec);
	MultiVector(const MultiVector & orig);

	std::size_t nvec() const {return mv.size();}

	void setSizeFromVector(const dolfin::GenericVector & v, std::size_t nvec);

	std::shared_ptr<const dolfin::GenericVector> operator[](std::size_t i) const;
	std::shared_ptr<dolfin::GenericVector> operator[](std::size_t i);

	// m[i] = this[i] \cdot v
	void dot(const dolfin::GenericVector & v, dolfin::Array<double> & m) const;

	// m[i,j] = this[i] \cdot other[j]
	void dot(const MultiVector & other, dolfin::Array<double> & m) const;

	// v += sum_i alpha[i]*this[i]
	void reduce(dolfin::GenericVector & v, const dolfin::Array<double> & alpha) const;

	void axpy(double a, const dolfin::GenericVector & y);
	void axpy(const dolfin::Array<double> & a, const MultiVector & y);

	// this[k] *= a
	void scale(std::size_t k, double a);

	// this[k] *= a[k]
	void scale(const dolfin::Array<double> & a);

	void zero();

	void norm_all(const std::string norm_type, dolfin::Array<double> & norms) const;

	void swap(MultiVector & other);

	~MultiVector();

private:
	void dot_self(dolfin::Array<double> & m) const;

	std::vector<std::shared_ptr<dolfin::GenericVector> > mv;
};

}
