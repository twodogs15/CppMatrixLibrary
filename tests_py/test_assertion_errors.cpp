/*! This file is part of EMatrix, the C++ matrix library distribution.
 *  This project is licensed under the terms of the MIT license. The full text
 *  of the license file can be found in LICENSE.
 */

/// \file

#include <iostream>
#include <climits>

#include "EMatrix.h"

using namespace ematrix;
using namespace std;

#ifndef THE_TYPE
#define THE_TYPE double
#endif


int main(void) {

/// [m0a] Matrix memory allocation/storage assignment.
#ifdef FAIL_M0_A
        //Matrix< THE_TYPE, 1, (2ul<<30)-1 > A;
        cout << (2ul<<THESIZE) << endl;
        Matrix< THE_TYPE, 1, ((2ul<<THESIZE) - (2ul<<9)) > A;
        cout << sizeof(A) << endl;
#endif

/// [m9] STL list initialize constructor.
/// Assertion fail for too many elements.
#ifdef FAIL_M9_A
    Matrix< THE_TYPE, 2, 3 > A = {1,2,3,4,5,6,7};
#endif

/// [m10] STL list initialize constructor.
/// Assertion fail for too many elements.
#ifdef FAIL_M10_A
    Matrix< THE_TYPE, 2, 3 > A;    
    A = {1,2,3,4,5,6,7};
#endif
    
    return(0);
}

