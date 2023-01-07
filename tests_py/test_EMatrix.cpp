#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include "EMatrix.h"

namespace py = pybind11;

typedef ematrix::Matrix<double, 2, 3> myMatrix;
typedef ematrix::Matrix<double, 2, 3> myMatrix;

myMatrix add(myMatrix& a, myMatrix& b) { return a+b; }
myMatrix sub(myMatrix& a, myMatrix& b) { return a-b; }
myMatrix neg(myMatrix& a) { return -a; }
myMatrix scalar_mult(double& a, myMatrix& b) { return a*b; }
myMatrix mult_scalar(myMatrix& a, double& b) { return a*b; }

template< typename T >
//void f(const py::array_t< T, py::array::c_style > &b) {
void f(const py::array &b) {
    py::buffer_info info = b.request();
    if (info.format != py::format_descriptor< T >::format() || info.ndim != 2) {
        throw std::runtime_error("Incompatible buffer format!");
    }
    //auto *v = new myMatrix;
    //memcpy(v->pIJ(), info.ptr, sizeof(double) * (size_t) (v->rows() * v->cols()));
    myMatrix v(reinterpret_cast< T* >(info.ptr));
    std::cout << v << std::endl;
    //return v;
}

PYBIND11_MODULE(test_EMatrix, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("f", &f< double >);
    m.def("add", &add);
    m.def("sub", &sub);
    m.def("neg", &neg);
    m.def("scalar_mult", &scalar_mult);
    m.def("mult_scalar", &mult_scalar);

    py::class_< myMatrix >(m, "myMatrix", py::buffer_protocol())
        //.def(py::init<py::ssize_t, py::ssize_t>())
        
        // Construct from a buffer
        .def(py::init([](const py::array &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<double>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new myMatrix;
            memcpy(v->pIJ(), info.ptr, sizeof(double) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &myMatrix::rows)
        .def("cols", &myMatrix::cols)

        // Bare bones interface
        .def("__getitem__",
             [](const myMatrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                 if (static_cast<size_t>(i.first) >= m.rows() || static_cast<size_t>(i.second) >= m.cols()) {
                     throw py::index_error();
                 }
                 return m[i.first][i.second];
             })
        .def("__setitem__",
             [](myMatrix &m, std::pair<py::ssize_t, py::ssize_t> i, float v) {
                 if (static_cast<size_t>(i.first) >= m.rows() || static_cast<size_t>(i.second) >= m.cols()) {
                     throw py::index_error();
                 }
                 m[i.first][i.second] = v;
             })

        // Provide buffer access
        .def_buffer([](myMatrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.pIJ(),                            /* Pointer to buffer */
                {m.rows(), m.cols()},               /* Buffer dimensions */
                {sizeof(double) * size_t(m.cols()), /* Strides (in bytes) for each index */
                 sizeof(double)});
        });

}
