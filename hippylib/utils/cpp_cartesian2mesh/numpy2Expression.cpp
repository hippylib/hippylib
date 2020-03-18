#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <vector>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Constant.h>


namespace py = pybind11;

class NumpyScalarExpression3D : public dolfin::Expression
{
public:
	NumpyScalarExpression3D() :
    Expression(),
    data(),
    h_x(1.),
	h_y(1.),
	h_z(1.),
	off_x(0.),
	off_y(0.),
	off_z(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int j_m(std::floor( (x[1]-off_y) /h_y));
    int k_m(std::floor( (x[2]-off_z) /h_z));

    int i_p(std::ceil( (x[0]-off_x) /h_x));
    int j_p(std::ceil( (x[1]-off_y) /h_y));
    int k_p(std::ceil( (x[2]-off_z) /h_z));

    const int i_max = data.shape(0);
    const int j_max = data.shape(1);
    const int k_max = data.shape(2);


    if (i_p < 0 || i_m >= i_max || j_p < 0 || j_m >= j_max || k_m < 0 || k_p >= k_max)
    {
    	values[0] = 0.;
    	return;
    }

    auto dd = data.unchecked<3>();

    if(i_m < 0)
    	i_m = 0;
    if(i_p >= i_max)
    	i_p = i_max-1;
    if(j_m < 0)
    	j_m = 0;
    if(j_p >= j_max)
    	j_p = j_max-1;
    if(k_m < 0)
    	k_m = 0;
    if(k_p >= k_max)
    	k_p = k_max-1;

    const double alpha_x = (x[0] - off_x)/h_x - static_cast<double>(i_m);
    const double alpha_y = (x[1] - off_y)/h_y - static_cast<double>(j_m);
    const double alpha_z = (x[2] - off_z)/h_z - static_cast<double>(k_m);


    values[0] = (1. - alpha_x)*(1. - alpha_y)*(1. - alpha_z)*dd(i_m, j_m, k_m) +
    			(1. - alpha_x)*(1. - alpha_y)*      alpha_z *dd(i_m, j_m, k_p) +
				(1. - alpha_x)*      alpha_y *(1. - alpha_z)*dd(i_m, j_p, k_m) +
				(1. - alpha_x)*      alpha_y *      alpha_z *dd(i_m, j_p, k_p) +
				      alpha_x *(1. - alpha_y)*(1. - alpha_z)*dd(i_p, j_m, k_m) +
				      alpha_x *(1. - alpha_y)*      alpha_z *dd(i_p, j_m, k_p) +
				      alpha_x *      alpha_y *(1. - alpha_z)*dd(i_p, j_p, k_m) +
				      alpha_x *      alpha_y *      alpha_z *dd(i_p, j_p, k_p);

  }

  void setData(py::array_t<double>& d, double & hh_x, double & hh_y, double & hh_z)
  {
	  data = d;
	  h_x = hh_x;
	  h_y = hh_y;
	  h_z = hh_z;
  }
  void setOffset(double & offset_x, double & offset_y, double & offset_z)
  {
	  off_x = offset_x;
	  off_y = offset_y;
	  off_z = offset_z;
  }

private:
  py::array_t<double> data;
  double h_x;
  double h_y;
  double h_z;
  double off_x;
  double off_y;
  double off_z;

};

class NumpyScalarExpression2D : public dolfin::Expression
{
public:
	NumpyScalarExpression2D() :
    Expression(),
    data(),
    h_x(1.),
	h_y(1.),
	off_x(0.),
	off_y(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int i_p(std::ceil( (x[0]-off_x) /h_x));

    int j_m(std::floor( (x[1]-off_y) /h_y));
    int j_p(std::ceil( (x[1]-off_y) /h_y));

    const int i_max = data.shape(0);
    const int j_max = data.shape(1);

    auto dd = data.unchecked<2>();

    if(i_p < 0 || i_m >= i_max || j_p < 0 || j_m >= j_max)
    {
    	values[0] = 0.;
    	return;
    }

    if(i_m < 0)
    	i_m = 0;
    if(i_p >= i_max)
    	i_p = i_max-1;
    if(j_m < 0)
    	j_m = 0;
    if(j_p >= j_max)
    	j_p = j_max-1;

    const double alpha_x = (x[0] - off_x)/h_x - static_cast<double>(i_m);
    const double alpha_y = (x[1] - off_y)/h_y - static_cast<double>(j_m);

    values[0] = (1. - alpha_x)*(1. - alpha_y)*dd(i_m, j_m) +
    		    (1. - alpha_x)*      alpha_y *dd(i_m, j_p) +
				      alpha_x *(1. - alpha_y)*dd(i_p, j_m) +
				      alpha_x *      alpha_y *dd(i_p, j_p);
  }

  void setData(py::array_t<double>& d, double & hh_x, double & hh_y)
  {
	  data = d;
	  h_x = hh_x;
	  h_y = hh_y;
  }
  void setOffset(double & offset_x, double & offset_y)
  {
	  off_x = offset_x;
	  off_y = offset_y;
  }

private:
  py::array_t<double> data;
  double h_x;
  double h_y;
  double off_x;
  double off_y;

};

class NumpyScalarExpression1D : public dolfin::Expression
{
public:
	NumpyScalarExpression1D() :
    Expression(),
    data(),
    h_x(1.),
	off_x(0.)
    {
    }

void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    int i_m(std::floor( (x[0]-off_x) /h_x));
    int i_p(std::ceil( (x[0]-off_x) /h_x));

    const int i_max = data.shape(0);
	auto dd = data.unchecked<1>();

    if(i_p < 0 || i_m > i_max)
    {
    	values[0] = 0.;
    	return;
    }

    if(i_m < 0)
    	i_m = 0;

    if(i_p > i_max)
    	i_p = i_max-1;

    const double alpha = (x[0] - off_x)/h_x - static_cast<double>(i_m);

    values[0] = (1. - alpha)*dd(i_m) + alpha*dd(i_p);
  }

  void setData(py::array_t<double>& d, double & hh_x)
  {
	  data = d;
	  h_x = hh_x;
  }
  void setOffset(double & offset_x)
  {
	  off_x = offset_x;
  }

private:
  py::array_t<double> data;
  double h_x;
  double off_x;

};
