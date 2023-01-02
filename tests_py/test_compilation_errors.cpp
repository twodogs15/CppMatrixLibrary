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

// [m0] virtual ~Matrix ();
#ifdef PASS_M0
    // Scoped to test ~Matrix(), also used gdb to verify the d'tor is called.
    // Matrix() and matalloc(size_t iRowIndex, size_t iColIndex) are
    // exercised as a side effect.  This would matter more for dynamic memory. 
    {
        Matrix< THE_TYPE, 2, 3 > A;
    }
#endif

// [m1] Matrix ();
// Testing a variety of compilation errors given invalid matrix sizes.
// Does not test an upper limit of matrix size, however;
#ifdef FAIL_M1_A
        Matrix< THE_TYPE, 0, 0 > A;
#endif

#ifdef FAIL_M1_B
        Matrix<double,0,1> A;
#endif

#ifdef FAIL_M1_C
        Matrix<double,1,0> A;
#endif

#ifdef FAIL_M1_D
        Matrix<double,-1,1> A;
#endif

#ifdef FAIL_M1_E
        Matrix<double,1,-1> A;
#endif

// [m2] Matrix (const Matrix< tData, tRows, tCols >& R);
// Testing a variety of compilation errors given invalid matrix sizes.
#ifdef FAIL_M2_A
        Matrix<double,2,3> A;
        Matrix<double,3,2> B(A);
#endif

#ifdef FAIL_M2_B
        Matrix<double,2,3> A;
        Matrix<double,3,2> B{A};
#endif

#ifdef FAIL_M2_C
// verified copy c'tor call and not the assignment operator.
        Matrix<double,2,3> A;
        Matrix<double,3,2> B=A;  
#endif
    
    return(0);
}

