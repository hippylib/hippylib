# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
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

from __future__ import absolute_import, division, print_function

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

code_AnisTensor2D = '''
class AnisTensor2D : public Expression
{
public:

  AnisTensor2D() :
  Expression(2,2),
  theta0(1.),
  theta1(1.),
  alpha(0)
  {

  }

void eval(Array<double>& values, const Array<double>& x) const
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
  
  double theta0;
  double theta1;
  double alpha;
  
};
'''

code_Mollifier = '''
class Mollifier : public Expression
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

void eval(Array<double>& values, const Array<double>& x) const
  {
        double sa = sin(alpha);
        double ca = cos(alpha);
        double c00 = theta0*sa*sa + theta1*ca*ca;
        double c01 = (theta0 - theta1)*sa*ca;
        double c11 = theta0*ca*ca + theta1*sa*sa;
        
        int ndim(2);
        Array<double> dx(ndim);
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
  
  double l;
  double o;
  
  double theta0;
  double theta1;
  double alpha;
  
  private:
    int nlocations;
    std::vector<double> locations;
  
};
'''
