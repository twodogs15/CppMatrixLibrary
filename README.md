# EMatrix - README

EMatrix: The lightweight matrix library developed in C++. 

EMatrix is designed to help solve engineering and math problems requiring 
vectors and matrices. EMatrix is operating system independent and as 
portable as can be made.  EMatrix supports overloaded operators, links with 
Lapack, and includes functions to facilitate porting from Octave type matrix 
scripting languages or Python/Numpy to C++.

## Platforms
EMatrix compiles and runs with g++ and clang++, as well as, Visual Studio 2015.

## Simple Example:
```c++
    #include <iostream>
    #include "EMatrix.h"

    int main(void) {

        ematrix::Matrix<double,3,3> A = {1.,2.,3.,0.,0.,0.,0,0,0};
        ematrix::Matrix<double,3,3> B = {1,1,1,3,4,5,3,6,10};
        std::cout << A+inv(B);

        ematrix::Matrix<double,2,2> C = {1,1,3,4};
        std::cout << C*inv(C);

        return(0);
    }
```
The above excerpt is from [readme_example.cpp](readme_example.cpp).  For a 
more complete list of examples of usage, available functions and available 
operators, see [test_EMartrix.cpp](test_EMatrix.cpp). EMatrix.h has usage 
comments for each method and operator as well.


## Compilation
### Using CMake
Using the **cmake system**, use the following commands:
```bash
$ cmake -B build && cd build && make
```

### Using the Command Line
**To compile:** Use the following line if you do not have Lapack
installed and you are only inverting 2x2 or 3x3 matrices.  
```bash
$ g++ readme_example.cpp
```
If you have Lapack installed, you can use one of the following:
```bash
$ g++ readme_example.cpp -I. -llapack -lblas 
$ g++ readme_example.cpp -I. "-l:liblapack.so.3"
```  

## Installation
Place EMatrix.h in the directory of your choosing.  Tell your 
compiler where it is. For g++, add -I\<your directory\>; see the provided 
Makefile for an example. In your code, #include "EMatrix.h" where you need to 
and off you go.

## Dependencies
If one needs to take a matrix inverse or determinant of only 2x2 or 3x3 
double precision numbers, then Lapack is *not* needed.  The functions are 
implemented as fully specialized methods in the templated header.  Actually, 
Python-Sympy was used to generate the code for the methods.  If one needs to 
generate additional inverses or determinants of single precision data types, 
complex data types, or higher dimensions, the Python code used to do so is 
included in the comments of EMatrix.h near the method of interest. 

Cramer's rule is used to for the implemented inverses and determinants.  As a
numerical solution, Cramer's rules is only practical for matrices of small 
dimension.  If inverses and determinants of larger matrix dimensions are 
required, the matrix library, Lapack, is recommended. 

Lapack is an open source linear algebra package and is available 
from http://www.netlib.org/lapack.  For Linux, Cygwin and MinGW users,  one can 
install or build Lapack very easily.  Visual Studio platforms can require a bit 
more effort.  Most expediently, one can download the compiled libraries from
http://icl.cs.utk.edu/lapack-for-windows/lapack. Another option is to compile and 
link with CLapack, an f2c'd version of Lapack.  If one does not want to compile 
the entire CLapack library, one can extract the necessary files and dependencies 
from CLapack to compile and link with EMatrix.h in less than two hours.      

## Authors
EMatrix, formerly CMatrix and DMatrix, was originally written by 
The Riddler.  This Author picked up the baton.  Many others helped with 
suggestions and bug fixes. 

Mail suggestions to [Dog House](mailto:twodogs15@yahoo.com)
 
## License
This file is part of EMatrix, the C++ matrix library distribution.
This project is licensed under the terms of the MIT license. The full text
of the license file can be found in [LICENSE](LICENSE).
