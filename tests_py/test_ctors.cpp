#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include "EMatrix.h"

namespace py = pybind11;

typedef ematrix::Matrix<double, 2, 3> myMatrix;

template< typename T >
py::buffer_info checkBufferType(const py::array_t< T, py::array::c_style > & a) {
    py::buffer_info info = a.request();
    if (info.format != py::format_descriptor< T >::format() || info.ndim != 2) {
        throw std::runtime_error("Incompatible buffer format!");
    }
    return info;
}

template< typename T >
myMatrix f(const py::array &b) {
    py::buffer_info info = b.request();
    if (info.format != py::format_descriptor< T >::format() || info.ndim != 2) {
        throw std::runtime_error("Incompatible buffer format!");
    }
    //auto *v = new myMatrix;
    //memcpy(v->pIJ(), info.ptr, sizeof(double) * (size_t) (v->rows() * v->cols()));
    myMatrix v(reinterpret_cast< T* >(info.ptr));
    std::cout << v << std::endl;
    return v;
}

/// [m2] Default constructor.
myMatrix test_ctor_m2(void) {
    // This is not the greatest test because we only really check that the
    // storage space is populated with zeros and that the rows and cols contain
    // the correct numbers.  The real work in down in matalloc.
    myMatrix A;
    return A;
}

/// [m3] Copy constructor.
template < typename T >
myMatrix test_ctor_m3(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    const myMatrix b;
    // Filling data manually to avoid using other functions.
    memcpy(b.pIJ(), info.ptr, sizeof(double)*(b.rows()*b.cols()));
    myMatrix c(b);
    return c;
}

/// [m4] Copy assignment operator.
template < typename T >
myMatrix test_ctor_m4(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    const myMatrix b;
    myMatrix c;
    // Filling data manually to avoid using other functions.
    memcpy(b.pIJ(), info.ptr, sizeof(double)*(b.rows()*b.cols()));
    c = b; 
    return c;
}

/// [m6b] Move constructor. Calls [m3] when DYNAMIC_STORAGE == 0
template < typename T >
myMatrix test_ctor_m6b(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    myMatrix b{};
    // Filling data manually to avoid using other functions.
    memcpy(b.pIJ(), info.ptr, sizeof(double)*(b.rows()*b.cols()));
    myMatrix c = std::move(b);
    return c;
}

/// [m7b] Move assignment operator. Calls [m4] when DYNAMIC_STORAGE == 0
template < typename T >
myMatrix test_ctor_m7b(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    myMatrix b, c;
    // Filling data manually to avoid using other functions.
    memcpy(b.pIJ(), info.ptr, sizeof(double)*(b.rows()*b.cols()));
    c = (b+b); // c = b; always called the move ctor.
    return c;
}

/// [m8] Memory, i.e. pointer or array, initialize constructor.
template < typename T >
myMatrix test_ctor_m8(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    myMatrix b(reinterpret_cast<T*>(info.ptr));
    return b;
}

/// [m9] STL list initialize constructor.
template < typename T >
myMatrix test_ctor_m9(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    T* p = reinterpret_cast<T*>(info.ptr);
    myMatrix b = {p[0], p[1], p[2], p[3], p[4], p[5]};
    // test_assertion.{cpp,py} tests too many elements.
    return b;
}

/// [m10] STL list initialize constructor.
template < typename T >
myMatrix test_ctor_m10(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    T* p = reinterpret_cast<T*>(info.ptr);
    myMatrix b;
    b = {p[0], p[1], p[2], p[3], p[4], p[5]};
    // test_assertion.{cpp,py} tests too many elements.
    return b;
}

/// [m11] Builtin C/C++ array copy to Matrix.
template < typename T >
myMatrix test_ctor_m11(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    T* p = reinterpret_cast<T*>(info.ptr);
    myMatrix b;
    b.load(p);
    return b;
}

/// [m12] memcpy to C-Style Array.
template < typename T >
myMatrix test_ctor_m12(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    myMatrix b(reinterpret_cast<T*>(info.ptr));
    T array[2][3] = {0};
    b.memcpy(&array[0][0]);
    myMatrix c(&array[0][0]);
    return c;
}

/// [m16] Submatrix assignment (n-1) based.
template < typename T >
myMatrix test_ctor_m16(const py::array_t< T, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    myMatrix b;
    ematrix::Matrix< T, 2, 2> c;
    // Filling data manually to avoid using other functions.
    memcpy(c.pIJ(), info.ptr, sizeof(T)*(c.rows()*c.cols()));
    b.submatrix(0,1,c);
    return b;
}

/// [m17] Submatrix extraction (n-1) based.
// Cannot use template here.  see comment below
myMatrix test_ctor_m17(const py::array_t< double, py::array::c_style > &a) {
    auto info = checkBufferType(a);
    
    // Cannot use a type template in the matrix below. the call to 
    // submatrix<y,y>() will not compile otherwise
    ematrix::Matrix< double, 3, 6 > c;
    memcpy(c.pIJ(), info.ptr, sizeof(double)*(c.rows()*c.cols()));

    myMatrix b;
    
    b = c.submatrix< 2, 3 >(0,2);
    return b;
}

PYBIND11_MODULE(test_ctors, m) {
    m.doc() = "pybind11 test_ctors";

    // m.def("f", &f< double >);
    m.def("test_ctor_m2",  &test_ctor_m2);
    m.def("test_ctor_m3",  &test_ctor_m3 < double >);
    m.def("test_ctor_m4",  &test_ctor_m4 < double >);
    m.def("test_ctor_m6b", &test_ctor_m6b< double >);
    m.def("test_ctor_m7b", &test_ctor_m7b< double >);
    m.def("test_ctor_m8",  &test_ctor_m8 < double >);
    m.def("test_ctor_m9",  &test_ctor_m9 < double >);
    m.def("test_ctor_m10", &test_ctor_m10< double >);
    m.def("test_ctor_m11", &test_ctor_m11< double >);
    m.def("test_ctor_m12", &test_ctor_m12< double >);
    m.def("test_ctor_m16", &test_ctor_m16< double >);
    m.def("test_ctor_m17", &test_ctor_m17);

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
