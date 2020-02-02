# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl

'''
This file contains C++ Expression to implement
- A symmetric tensor in 2D of the form:
  [  sin(alpha)  cos(alpha) ] [theta0    0   ] [ sin(alpha)  -cos(alpha) ]
  [ -cos(alpha)  sin(alpha) ] [   0   theta1 ] [ cos(alpha)   sin(alpha) ]
- A mollifier function f of the form:
  f(x) = \sum_{i} exp( -|| x - x_i ||^o_B/l^o ),
  where:
  - x_i (i = i,...,n) are given locations in space
  - o                 is the order of the mollifier
  - l                 is a correlation lenght
  - B                 is a s.p.d. tensor in 2D as above.
'''

cpp_code = '''
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
#include <vector>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Constant.h>
class AnisTensor2D : public dolfin::Expression
{
public:
  AnisTensor2D() :
      Expression(2,2),
      theta0(1.),
      theta1(1.),
      alpha(0)
      {
      }
      
    friend class Mollifier;
void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
     double sa = sin(alpha);
     double ca = cos(alpha);
     double c00 = theta0*sa*sa + theta1*ca*ca;
     double c01 = (theta0 - theta1)*sa*ca;
     double c11 = theta0*ca*ca + theta1*sa*sa;
  
     values[0] = c00;
     values[1] = c01;
     values[2] = c01;
     values[3] = c11;
  }
  
  void set(double _theta0, double _theta1, double _alpha)
  {
  theta0 = _theta0;
  theta1 = _theta1;
  alpha  = _alpha;
  }
  
private:
  double theta0;
  double theta1;
  double alpha;
  
};
class Mollifier : public dolfin::Expression
{
public:
  Mollifier() :
  Expression(),
  nlocations(0),
  locations(nlocations),
  l(1),
  o(2),
  theta0(1),
  theta1(1),
  alpha(0)
  {
  }
void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
        double sa = sin(alpha);
        double ca = cos(alpha);
        double c00 = theta0*sa*sa + theta1*ca*ca;
        double c01 = (theta0 - theta1)*sa*ca;
        double c11 = theta0*ca*ca + theta1*sa*sa;
        
        int ndim(2);
        Eigen::VectorXd dx(ndim);
        double e(0), val(0);
        for(int ip = 0; ip < nlocations; ++ip)
        {
            for(int idim = 0; idim < ndim; ++idim)
                dx[idim] = x[idim] - locations[2*ip+idim];
                
            e = dx[0]*dx[0]*c00 + dx[1]*dx[1]*c11 + 2*dx[0]*dx[1]*c01;
            val += exp( -pow(e/(l*l), .5*o) );
        }
        values[0] = val;
  }
  
  void addLocation(double x, double y) { locations.push_back(x); locations.push_back(y); ++nlocations;}
  
  void set(const AnisTensor2D & A, double _l, double _o)
  {
    theta0 = 1./A.theta0;
    theta1 = 1./A.theta1;
    alpha = A.alpha;
    
    l = _l;
    o = _o;
  }
    
  private:
    double l;
    double o;
  
    double theta0;
    double theta1;
    double alpha;
    
    int nlocations;
    std::vector<double> locations;
  
};
PYBIND11_MODULE(SIGNATURE, m)
    {
    py::class_<AnisTensor2D, std::shared_ptr<AnisTensor2D>, dolfin::Expression>
    (m, "AnisTensor2D")
    .def(py::init<>())
    .def("set", &AnisTensor2D::set);
    
        py::class_<Mollifier, std::shared_ptr<Mollifier>, dolfin::Expression>
    (m, "Mollifier")
    .def(py::init<>())
    .def("set", &Mollifier::set)
    .def("addLocation", &Mollifier::addLocation);
    
    }
'''

ExpressionModule = dl.compile_cpp_code(cpp_code)