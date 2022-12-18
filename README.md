# EMatrix - README

EMatrix: The lightweight matrix library developed in C++. 

EMatrix is designed to help solve engineering and math problems requiring 
vectors and matrices. EMatrix is OS independent and is as portable as we can 
make it.  EMatrix supports overloaded operators, links with Lapack, and 
includes functions to facilitate porting from Octave type matrix scripting 
languages to C++.

## Platforms
EMatrix compiles and runs with g++ and clang++, as well as, Visual Studio 2015.

## Simple Example:

    #include <iostream>
    #include "EMatrix.h"

    int main(void) {

        ematrix::Matrix<double,3,3> A = {1.,2.,3.,0.,0.,0.,0,0,0};
        ematrix::Matrix<double,3,3> B = {1,1,1,3,4,5,3,6,10};
        std::cout << A+inv(B);

        return(0);
    }

**To compile:** $ g++ example_in_readme.cpp -llapack -lblas or 
g++ example_in_readme.cpp "-l:liblapack.so.3"

See also [test_EMartrix.cpp](test_EMatrix.cpp) for an additional ideas.  

## Installation
Place EMatrix.h in the directory of your choosing.  Tell your 
compiler where it is. For g++, add -I<your directory>; see the provided 
Makefile for an example. In your code, #include "EMatrix.h" where you need to 
and off you go.

## Dependencies
If one needs to take a matrix inverse or determinant, Lapack 
is required.  Lapack is an open source linear algebra package and is available 
from http://www.netlib.org/lapack.  For Linux, Cygwin and MinGW users,  one can 
install or build Lapack very easily.  Visual Studio platforms can require a bit 
more effort.  Most expediently, one can download the compiled libraries from
http://icl.cs.utk.edu/lapack-for-windows/lapack. Another option is compile and 
link with CLapack, and f2c'd version of Lapack.  In fact, one can extract the 
necessary files in less than two hours.  In future versions, I plan to include 
the requisite files.  I also plan to investigate the now included C interface
to Lapack.      

## Authors
EMatrix, formerly CMatrix and DMatrix, was originally written by 
The Riddler.  This Author picked up the baton.  Many others helped with 
suggestions and bug fixes. 

Mail suggestions to [Dog House](mailto:twodogs15@yahoo.com)
 
## License
This file is part of EMatrix, the C++ matrix library distribution.
This project is licensed under the terms of the MIT license. The full text
of the license file can be found in [LICENSE](LICENSE).

