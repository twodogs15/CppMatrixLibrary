/*! This file is part of EMatrix, the C++ matrix library distribution.
 *  This project is licensed under the terms of the MIT license. The full text
 *  of the license file can be found in LICENSE.                         
 */

/// \file

#include <iostream>

#include "EMatrix.h"

using namespace ematrix;
using namespace std;

#ifndef THE_TYPE 
#define THE_TYPE double
#endif

int main(void) {
//! Begin and the very beginning.

    // Scoped to test [virtual ~Matrix ()], used gdb and cerr to verify
    {
        // Matrix ();
        // void matalloc (size_t iRowIndex, size_t iColIndex);
        Matrix< THE_TYPE, 2, 3 > A;

        // These invoke compiler errors
#ifdef FAIL_01
        Matrix< THE_TYPE, 0, 0 > A;
#endif

#ifdef FAIL_02
        Matrix<double,0,1> A;
#endif

#ifdef FAIL_03
        Matrix<double,1,0> A;
#endif
    }  
    return(0);
}

