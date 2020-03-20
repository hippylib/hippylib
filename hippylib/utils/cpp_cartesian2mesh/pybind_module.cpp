PYBIND11_MODULE(SIGNATURE, m)
{
	m.def("numpy2MeshFunction",  &numpy2MeshFunction,
		  "Convert an ND numpy array to MeshFunction",
		  py::arg("mesh"),
		  py::arg("h"),
		  py::arg("data"),
		  py::arg("mfun"));

    py::class_<NumpyScalarExpression3D, std::shared_ptr<NumpyScalarExpression3D>, dolfin::Expression>
    (m, "NumpyScalarExpression3D")
    .def(py::init<>(),"Interpolate a 3D numpy array on the mesh - trilinear interpolation")
    .def("setData", &NumpyScalarExpression3D::setData, py::arg("data"), py::arg("h_x"), py::arg("h_y"), py::arg("h_z"))
	.def("setOffset", &NumpyScalarExpression3D::setOffset, py::arg("offset_x"), py::arg("offset_y"), py::arg("offset_z"));

    py::class_<NumpyScalarExpression2D, std::shared_ptr<NumpyScalarExpression2D>, dolfin::Expression>
    (m, "NumpyScalarExpression2D", "Interpolate a 2D numpy array on the mesh - bilinear interpolation")
    .def(py::init<>())
    .def("setData", &NumpyScalarExpression2D::setData, py::arg("data"), py::arg("h_x"), py::arg("h_y"))
	.def("setOffset", &NumpyScalarExpression2D::setOffset, py::arg("offset_x"), py::arg("offset_y"));

    py::class_<NumpyScalarExpression1D, std::shared_ptr<NumpyScalarExpression1D>, dolfin::Expression>
    (m, "NumpyScalarExpression1D", "Interpolate a 1D numpy array on the mesh - linear interpolation")
    .def(py::init<>())
    .def("setData", &NumpyScalarExpression1D::setData, py::arg("data"), py::arg("h_x"))
	.def("setOffset", &NumpyScalarExpression1D::setOffset, py::arg("offset_x"));
}
