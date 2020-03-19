/*
 * U. Villa
 */

#include <cmath>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshFunction.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace dl = dolfin;

void numpy2MeshFunction3D(dl::Mesh & mesh,
						double h_x,
						double h_y,
						double h_z,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	int i,j,k;
	auto dd = data.unchecked<3>();

	const int i_max = data.shape(0);
	const int j_max = data.shape(1);
	const int k_max = data.shape(2);

	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor(p.x()/h_x));
        j = int(std::floor(p.y()/h_y));
        k = int(std::floor(p.z()/h_z));

        if(i<0)
        	i=0;
        if(i >= i_max)
        	i = i_max-1;
        if(j<0)
        	j=0;
        if(j >= j_max)
        	j = j_max-1;
        if(k<0)
        	k=0;
        if(k >= k_max)
        	k = k_max-1;

        mfun[*cell] = dd(i,j,k);
	}
}


void numpy2MeshFunction2D(dl::Mesh & mesh,
						double h_x,
						double h_y,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	int i,j;

	auto dd = data.unchecked<2>();

	const int i_max = data.shape(0);
	const int j_max = data.shape(1);

	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor(p.x()/h_x));
        j = int(std::floor(p.y()/h_y));

        if(i<0)
        	i=0;
        if(i >= i_max)
        	i = i_max-1;
        if(j<0)
        	j=0;
        if(j >= j_max)
        	j = j_max-1;

        mfun[*cell] = dd(i,j);
	}
}

void numpy2MeshFunction1D(dl::Mesh & mesh,
						double h,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	int i;
	auto dd = data.unchecked<1>();

	const int i_max = data.shape(0);
	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor(p.x()/h));

        if(i<0)
        	i=0;
        if(i >= i_max)
        	i = i_max-1;

        mfun[*cell] = dd(i);
	}
}

void numpy2MeshFunction(dl::Mesh & mesh,
						py::array_t<double> & h,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	auto hh = h.unchecked<1>();
	if(mesh.geometry().dim() == 3)
		numpy2MeshFunction3D(mesh, hh(0), hh(1), hh(2), data, mfun);
	else if (mesh.geometry().dim() == 2)
		numpy2MeshFunction2D(mesh, hh(0), hh(1), data, mfun);
	else
		numpy2MeshFunction1D(mesh, hh(0), data, mfun);
}
