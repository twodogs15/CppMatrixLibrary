/*! This file is part of EMatrix, the C++ matrix library distribution.
 *  This project is licensed under the terms of the MIT license. The full text
 *  of the license file can be found in LICENSE.                         
 */

/// \file


// to compile: g++ example_in_readme.cpp -llapack -lblas
// or          g++ example_in_readme.cpp "-l:liblapack.so.3"

#include <iostream>
#include "EMatrix.h"

//! Begin and the very beginning.
int main(void) {

    ematrix::Matrix<double,3,3> A = {1.,2.,3.,0.,0.,0.,0,0,0};
    ematrix::Matrix<double,3,3> B = {1,1,1,3,4,5,3,6,10};
    std::cout << A+inv(B);

    ematrix::Matrix<double,2,2> C = {1,1,3,4};
    std::cout << C*inv(C);

    return(0);
}

/* Those interested in generating integer matrix inverse pairs, please see:
 * Ericksen, W. S.  Inverse Pairs of Matrices with Integer Elements. 
 * SIAM Journal on Numerical Analysis, 1980, Vol. 17, No. 3 : pp. 474-477 
 */
