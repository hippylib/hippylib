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
						double h,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	int i,j,k;
	auto dd = data.unchecked<3>();
	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor(p.x()/h));
        j = int(std::floor(p.y()/h));
        k = int(std::floor(p.z()/h));

        if (i < 0 || i >= data.shape(0) || j < 0 || j >= data.shape(1) || k < 0 || k >= data.shape(2))
        	mfun[*cell] = 0.;
        else
        	mfun[*cell] = dd(i,j,k);
	}
}


void numpy2MeshFunction2D(dl::Mesh & mesh,
						double h,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	int i,j;
	auto dd = data.unchecked<2>();
	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor(p.x()/h));
        j = int(std::floor(p.y()/h));

        if (i < 0 || i >= data.shape(0) || j < 0 || j >= data.shape(1))
        	mfun[*cell] = 0.;
        else
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
	for (dl::CellIterator cell(mesh); !cell.end(); ++cell)
	{
		dl::Point p = cell->midpoint();
        i = int(std::floor(p.x()/h));

        if (i < 0 || i >= data.shape(0) )
        	mfun[*cell] = 0.;
        else
        	mfun[*cell] = dd(i);
	}
}

void numpy2MeshFunction(dl::Mesh & mesh,
						double h,
						py::array_t<uint> & data,
						dl::MeshFunction<std::size_t> & mfun)
{
	if(mesh.geometry().dim() == 3)
		numpy2MeshFunction3D(mesh, h, data, mfun);
	else if (mesh.geometry().dim() == 2)
		numpy2MeshFunction2D(mesh, h, data, mfun);
	else
		numpy2MeshFunction1D(mesh, h, data, mfun);
}
