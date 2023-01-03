/** This file is part of EMatrix, the C++ matrix library distribution.
 *  This project is licensed under the terms of the MIT license. The full text
 *  of the license file can be found in LICENSE.
 */

/// \file

#ifndef _EMATRIX_H
#define _EMATRIX_H 1

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cassert>

#include <complex>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <random>
#include <limits>
#include <algorithm>

#ifdef TESTING
#define HERE(x) std::cerr << (x) << ':' << __FILE__ << ':' << __LINE__ << std::endl;
#else
#define HERE(x)
#endif

namespace ematrix {

#define STACK_STORAGE 1

#if STACK_STORAGE
#define BEGIN(x) std::begin(x)
#define END(x) std::end(x)
#else
#define BEGIN(x) (x)
#define END(x) (x+tRows*tCols)
#endif

template < typename tData, size_t tRows, size_t tCols >
class Matrix {
  protected:
    /// Matrix memory allocation/storage assignment
    void matalloc (size_t iRowIndex, size_t iColIndex);
    
    /// Number of row and columns
    size_t iRows, iCols;
    
    /// Storage elements.
    /// matalloc above assigns an array of pointers to pointers
    tData* ij[tRows];
    tData storage[tRows*tCols];

  public:
    // Numeric labeling m1 - m? are identifiers to help keep track of testing

    /// [m0] Virtual destructor.
    virtual ~Matrix ();

    /** [m1] Default constructor.
     *  Usage: Matrix<double,2,3> A;
     */
    Matrix ();

    /** [m2] Copy constructor (not to be confused with the assignment operator).
     *  Usage: Matrix<float,2,3> A;
     *         Matrix<float,2,3> B=A;
     */
    Matrix (const Matrix< tData, tRows, tCols >& R);

    /** Memory, i.e. pointer or array, initialize constructor.
     *  Usage: float a[2][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0}};
     *         Matrix<float,2,3> A(&a[0][0]);
     */
    Matrix (tData* tArray);

    /** STL list initialize constructor (C++11).
     *  Usage: Matrix<double,3,3> A = {1.0,2.0,3.0,0.0,0.0,0.0,0,0,0};
     *         Numerical typing is lose as seen above.
     */
    Matrix (const std::initializer_list<tData>& l);

    /** Assignment operator (not to be confused with the copy constructor).
     *  Usage: C = A - B;
     */
    const Matrix< tData, tRows, tCols >& operator = (const Matrix< tData, tRows, tCols >& R);

    /** STL initializer_list assignment (C++11).
     *  Usage: Matrix<double,3,2> A;
     *         A = {1.1,2.1,3.1,0.0,0.0,0.0};
     */
    const Matrix< tData, tRows, tCols >& operator = (const std::initializer_list<tData>& l);

    /** Builtin C/C++ array copy to Matrix.
     *  Usage: double a[2][3] = {{1.1,2.1,3.1},{4.0,5.0,6.0}};
     *         Matrix<double,2,3> A;
     *         A.load(&a[0][0]);
     *  Warning: Not cognizant of size, can read from unintended memory location,
     *           but does use the safer c++ idiomatic copy.
     */
    void memset(const tData* tArray);

    /** C/C++ like element access (0 to n-1), get and set.
     *  Usage: Matrix<double,3,2> A = {1,2,3,4,5,6};
     *         A[0][0] = 7;
     *         cerr << A[2][1] << endl;
     *  Row operator returns the matrix row corresponding to iRowIndex,
     *  Warning: Does not provide column access safety.
     */
    tData* operator [] (size_t iRowIndex) const {
        assert(iRowIndex<iRows);
        return ij[iRowIndex];
    }

    /** Data access operator for Octave and FORTRAN indexing (1 to n).
     *  Note this does not imply FORTRAN memory storage
     *  Usage: Matrix<double,3,2> A = {1,2,3,4,5,6};
     *         A(1,1) = 8;
     *         cerr << A(3,2) << endl;
     */
    tData& operator () (size_t iRowIndex, size_t iColIndex) const;

    /** Vector Data access operator for Octave and FORTRAN indexing (1 to n).
     *  Usage: Matrix<double,6,1> V = {1,2,3,4,5,6}; V(1) = 8; cerr << V(6) << endl;
     *  Usage: Matrix<double,1,6> U = {1,2,3,4,5,6}; U(1) = 8; cerr << U(6) << endl;
     *  Note a non 1xn or nx1 matrix will assertion fail.  Could not determine a way
     *  to force compile time error.
     */
    tData& operator () (size_t iIndex) const;

    /** Overloaded output stream operator <<.
     *  Usage: log_file << A;
     */
    template < class tData0, size_t tRows0, size_t tCols0 >
    friend std::ostream& operator << (std::ostream& s,const Matrix< tData0, tRows0, tCols0 >& A);

    /** Get the storage pointer for the data in the matrix.
     *  This is really only here for the friend functions
     *  Try not to use it
     *  Usage: tData* ptr = A.pIJ();
     */
    tData* pIJ (void) const {
        return ij[0];
    }

    /** Get the number of rows in a matrix.
     *  Usage: size_t i = A.rows();
     */
    size_t rows (void) const {
        return iRows;
    }

    /** Get the number of cols in a matrix.
     *  Usage: size_t i = A.cols();
     */
    size_t cols (void) const {
        return iCols;
    }

    /** Boolean == operator.
     *  Usage: if (A == B) ...
     */
    bool operator == (Matrix< tData, tRows, tCols >& R);

    /** Boolean != operator.
     *  Usage: if (A != B) ...
     */
    bool operator != (Matrix< tData, tRows, tCols >& R);

    /** Unary + operator.
     *  Usage: C = (+B); Just returns *this;
     */
    Matrix< tData, tRows, tCols > operator + ();

    /** Unary - operator.
     *  Usage: C = (-A);
     */
    Matrix< tData, tRows, tCols > operator - ();

    /** Addition operator.
     *  Usage: C = A + B;
     */
    Matrix< tData, tRows, tCols > operator + (const Matrix< tData, tRows, tCols >& R);

    /** Subtraction operator.
     *  Usage: C = A - B;
     */
    Matrix< tData, tRows, tCols > operator - (const Matrix< tData, tRows, tCols >& R);

    /** Scalar multiplication operator.
     *  Usage: C = A * scalar;
     */
    Matrix< tData, tRows, tCols > operator * (const tData& scalar);

    /** Friend scalar multiplication operator.
     *  Usage: C = scalar * A;
     */
    template< class tData0, size_t tRows0, size_t tCols0 >
    friend Matrix< tData0, tRows0, tCols0 > operator * (const tData0& scalar,const Matrix< tData0, tRows0, tCols0 >& R);

    /** Scalar division operator.
     *  Usage: C = A / scalar;
     */
    Matrix< tData, tRows, tCols > operator / (const tData& scalar);

    /** Matrix multiplication operator.
     *  Usage: C = A * B;
     *  Note that the full implementation is here in the class declaration
     *  to support template compiler checking of dimensions.
     */
    template < size_t tColsR >
    Matrix< tData, tRows, tColsR > operator * (const Matrix< tData, tCols, tColsR >& R) {
        tData x;
        Matrix< tData, tRows, tColsR > Result;

        for (size_t iIndex=0; iIndex<iRows; iIndex++) {
            for (size_t jIndex=0; jIndex<R.cols(); jIndex++) {
                x = tData(0);
                for (size_t kIndex=0; kIndex<R.rows(); kIndex++) {
                    x += ij[iIndex][kIndex] * R[kIndex][jIndex];
                }
                Result[iIndex][jIndex] = x;
            }
        }
        return Result;
    }

    /** Array multiplication operator.
     *  Usage: C = (A *= B); Must use parentheses
     *  This mimics Octave's A .* B operator and not C's x *= 5 operator
     */
    Matrix< tData, tRows, tCols > operator *= (const Matrix< tData, tRows, tCols >& R);

    /** Array division operator.
     *  Usage: C = (A /= B); Must use parentheses
     *  This mimics Octave's A ./ B operator and not C's x /= 5 operator
     */
    Matrix< tData, tRows, tCols > operator /= (const Matrix< tData, tRows, tCols >& R);

    /** Concatenate matrices top and bottom.
     *  Usage: C = (A & B); Must use parentheses
    */
    template < class tData0, size_t tCols0, size_t tRowsT, size_t tRowsB>
    friend Matrix< tData0, tRowsT+tRowsB, tCols0 > operator & (const Matrix< tData0, tRowsT, tCols0 >& Top,
            const Matrix< tData0, tRowsB, tCols0 >& Bottom);

    /** Concatenate matrices left to right.
    *   Usage: C = (A | B); Must use parentheses
    */
    template < class tData0, size_t tRows0, size_t tColsL, size_t tColsR >
    friend Matrix< tData0, tRows0, tColsL+tColsR > operator | (const Matrix< tData0, tRows0, tColsL >& Left,
            const Matrix< tData0, tRows0, tColsR >& Right);

    /** Set contents to 0x0.
     *  Usage: A.zeros();
     */
    Matrix< tData, tRows, tCols > zeros(void);

    /** Set contents to tData(1).
     *  Usage: A.ones();
     */
    Matrix< tData, tRows, tCols > ones(void);

    /** Set contents to the identity matrix.
     *  Usage: A.eye();
     */
    Matrix< tData, tRows, tCols > eye(void);

    /** Set all elements of the current matrix random ~N(0,1);
     *  Usage: u.randn();
     */
    Matrix< tData, tRows, tCols > randn(void);

    /** Matrix transpose and complex conjugate transpose.
     *  Usage: A_trans = trans(A);
     */
    template < class tData0, size_t tRows0, size_t tCols0 >
    friend Matrix< tData0, tCols0, tRows0 > trans(const Matrix< tData0, tRows0, tCols0 >& R);

    template < size_t tRows0, size_t tCols0 >
    friend Matrix< std::complex<float>, tCols0, tRows0 > trans(const Matrix< std::complex<float>, tRows0, tCols0 >& R);

    template < size_t tRows0, size_t tCols0 >
    friend Matrix< std::complex<double>, tCols0, tRows0 > trans(const Matrix< std::complex<double>, tRows0, tCols0 >& R);

    /** Matrix diagonal like Octave.
     *  This friend function does not modify input contents.
     *  Usage: A_diagVectorNx1 = diag(A);
     *         A_diagMatrixNxN = diag(A_diagVectorNx1);
     *         A_diagMatrixMxM = diag(A_diagVector1xM)
     */
    template < class tData0, size_t tRows0 >
    friend Matrix< tData0, tRows0, 1 > diag(const Matrix< tData0, tRows0, tRows0 >& R);

    template < class tData0, size_t tRows0 >
    friend Matrix< tData0, tRows0, tRows0 > diag(const Matrix< tData0, tRows0, 1 >& R);

    template < class tData0, size_t tCols0 >
    friend Matrix< tData0, tCols0, tCols0 > diag(const Matrix< tData0, 1, tCols0 >& R);

    /* Construct a skew symmetric matrix from a 3x1 vector.
     * w = [wx;wy;wz];
     * skew(w) = [0.0 -wz +wy
     *            +wz 0.0 -wx
     *            -wy +wx 0.0];
     * Usage: omega_x = skew(w);
     */
    template < class tData0 >
    friend Matrix< tData0, 3, 3 > skew(const Matrix< tData0, 3, 1 >& R);

    /* Take the cross product of two 3x1 vectors.
     * Usage: z = cross(x, y);
     */
    template < class tData0 >
    friend Matrix< tData0, 3, 1 > cross(const Matrix< tData0, 3, 1 >& L, const Matrix< tData0, 3, 1 >& R);

    /* Take the dot product of two 3x1 vectors.
     * Usage: (norm(x))^2 = dot(x, x);
     */
    template < class tData0, size_t tRows0 >
    friend tData0 dot(const Matrix< tData0, tRows0, 1 >& L, const Matrix< tData0, tRows0, 1 >& R);

    /** Take the norm of two vectors
     *  Usage: norm_a = a.n();
     */
    tData n(void) const;

    /** Take the norm of two vectors.
     *  Usage: norm_a = norm(a);
     */
    template < class tData0, size_t tRows0 >
    friend tData0 norm(const Matrix< tData0, tRows0, 1 >& R);

    /** return a unit vector in the direction of V.
     *  Usage: u_v = V.u()
     */
    Matrix< tData, tRows, 1 > u(void) const;

    /** Matrix inverse, must link with Lapack.
     *  Usage: A_inv = inv(A);
     */
    template < size_t tRows0 >
    friend Matrix< float, tRows0, tRows0 > inv(const Matrix< float, tRows0, tRows0 >& R);

    template < size_t tRows0 >
    friend Matrix< std::complex<float>, tRows0, tRows0 > inv(const Matrix< std::complex<float>, tRows0, tRows0 >& R);

    template < size_t tRows0 >
    friend Matrix< double, tRows0, tRows0 > inv(const Matrix< double, tRows0, tRows0 >& R);

    template < size_t tRows0 >
    friend Matrix< std::complex<double>, tRows0, tRows0 > inv(const Matrix< std::complex<double>, tRows0, tRows0 >& R);

    /** Matrix inverse, 2x2 and 3x3 double specializations.
     *  Usage A_inv = inv(A)
     */
    friend Matrix< double, static_cast<size_t>(2), static_cast<size_t>(2) >
    inv(const Matrix< double, static_cast<size_t>(2), static_cast<size_t>(2) >& R);

    friend Matrix< double, static_cast<size_t>(3), static_cast<size_t>(3) >
    inv(const Matrix< double, static_cast<size_t>(3), static_cast<size_t>(3) >& R);

    /** Matrix determinant, must link with Lapack.
     *  Usage: A_det = det(A);
     */
    template < size_t tRows0 >
    friend float det(const Matrix< float, tRows0, tRows0 >& R);

    template < size_t tRows0 >
    friend double det(const Matrix< double, tRows0, tRows0 >& R);

    template < size_t tRows0 >
    friend std::complex<float> det(const Matrix< std::complex<float>, tRows0, tRows0 >& R);

    template < size_t tRows0 >
    friend std::complex<double> det(const Matrix< std::complex<double>, tRows0, tRows0 >& R);

    /** Matrix determinant, 2x2 and 3x3 double specializations.
     *  Usage A_det = det(A)
     */
    friend double det(const Matrix< double, static_cast<size_t>(2), static_cast<size_t>(2) >& R);

    friend double det(const Matrix< double, static_cast<size_t>(3), static_cast<size_t>(3) >& R);

};  // EOC: End of Class


template < class tData, size_t tRows, size_t tCols >
void Matrix< tData, tRows, tCols >::matalloc(size_t iRowIndex, size_t iColIndex) {
    ij[0] = &storage[0];
    for (size_t iIndex = 1; iIndex < iRowIndex; iIndex++)
        ij[iIndex] = ij[iIndex - 1] + iColIndex;

    //std::memset (storage, 0x0, sizeof(storage));
    std::fill(BEGIN(storage), END(storage), tData(0));
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols >::~Matrix() {
    HERE("[m0] ~Matrix()");
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols >::Matrix()
    : iRows(tRows), iCols(tCols) {
    matalloc(tRows,tCols);

    // On some systems, 2 can be much faster that 1 (30x) but we will favor
    // more idiomatic C++ until such time as speed is important.  Also, memset
    // is not correct when using non built-in types, e.g. complex<double>.
    // 1. std::memset (storage, 0x0, sizeof(storage));
    // 2. std::memset (reinterpret_cast<char*>(&storage[0]), '\0', sizeof(storage));
    std::fill(BEGIN(storage), END(storage), tData(0));
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols >::Matrix(const Matrix & R)
    : iRows (R.iRows), iCols (R.iCols) {
    matalloc(R.iRows, R.iCols);
    // Choosing idiomatic C++ over potential speed
    // std::memcpy(storage, R.storage, sizeof(storage));
    std::copy(BEGIN(R.storage), END(R.storage), BEGIN(storage));
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols >::Matrix (tData* tArray)
    : iRows (tRows), iCols (tCols) {
    matalloc(tRows,tCols);
    // Choosing idiomatic C++ over potential speed
    // std::memcpy(storage, tArray, sizeof(storage));
    std::copy(tArray, tArray+tRows*tCols, BEGIN(storage));
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols >::Matrix (const std::initializer_list<tData>& l)
    : iRows(tRows), iCols(tCols) {
    assert( iRows*iCols == l.size() );
    matalloc( iRows, iCols);
    for(size_t i = 0; i<(iRows*iCols); i++) {
        ij[0][i] = *(l.begin() + i);
    }
}

template < class tData, size_t tRows, size_t tCols >
const Matrix< tData, tRows, tCols > & Matrix< tData, tRows, tCols >::operator = (const Matrix< tData, tRows, tCols > & R) {
    assert((iRows == R.iRows) && (iCols == R.iCols));
    if( this != &R ) {
        tData* tArray = R.ij[0];
        // std::memcpy (ij[0], R.ij[0], sizeof(tData)*iRows*iCols);
        std::copy(tArray, tArray+tRows*tCols, BEGIN(storage));
    }
    return *this;
}

template < class tData, size_t tRows, size_t tCols >
const Matrix< tData, tRows, tCols>& Matrix< tData,tRows,tCols>::operator = (const std::initializer_list<tData>& l) {
    assert( iRows*iCols == l.size() );
    for(size_t i = 0; i<(iRows*iCols); i++) {
        ij[0][i] = *(l.begin() + i);
    }
    return *this;
}

template < class tData, size_t tRows, size_t tCols >
void Matrix< tData, tRows, tCols >::memset(const tData* tArray) {
    // std::memcpy(storage, tArray, sizeof(storage));
    std::copy(tArray, tArray+tRows*tCols, BEGIN(storage));
}

template < class tData, size_t tRows, size_t tCols >
tData & Matrix< tData, tRows, tCols >::operator () (size_t iRowIndex, size_t iColIndex) const {
    iRowIndex--;
    iColIndex--;
    assert(iRowIndex<iRows);
    assert(iColIndex<iCols);
    return ij[iRowIndex][iColIndex];
}

template < class tData, size_t tRows, size_t tCols >
tData & Matrix< tData, tRows, tCols >::operator () (size_t iIndex) const {
    iIndex--;
    assert(iRows==1 || iCols==1);
    if(iCols == 1) {
        assert(iIndex<iRows);
        return ij[iIndex][0];
    } else { // (iRows == 1)
        assert(iIndex<iCols);
        return ij[0][iIndex];
    }
}

template < class tData, size_t tRows, size_t tCols >
std::ostream& operator << (std::ostream& s,const Matrix< tData, tRows, tCols >& A) {
    // Sets new precision, returns old. Should figure out how to modify
    // without code change here.  Switch to exponential as well.
    std::streamsize old_precision = s.precision(8);

    //s.setf( std::ios::fixed, std:: ios::floatfield ); // floatfield set to fixed

    //s << "Address: 0x" << (&A) << std::endl;
    for (size_t i=0; i<A.iRows; i++) {
        for (size_t j=0; j<A.iCols; j++) {
            //s.width(25);
            s << (A[i][j]) << '\t';
        }
        s << std::endl;
    }
    s.flush();
    s.precision(old_precision);
    return s;
}

template < class tData, size_t tRows, size_t tCols >
bool Matrix< tData, tRows, tCols >::operator == (Matrix< tData, tRows, tCols > & R) {
    tData * pLeft  = ij[0];
    tData * pRight = R.ij[0];

    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++)
        if((*pLeft++) != tData(*pRight++)) {
            return false;
        }

    return true;
}

template < class tData, size_t tRows, size_t tCols >
bool Matrix< tData, tRows, tCols >::operator != (Matrix< tData, tRows, tCols > & R) {
    return !(*this == R);
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator + () {
    return *this;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator - () { 
    Matrix< tData, tRows, tCols > Result;
    tData * pLeft = ij[0];
    tData * pResult = Result.ij[0];
    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pResult++) = (-(*pLeft++));
    return Result;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator + (const Matrix< tData, tRows, tCols > & R) {
    Matrix< tData, tRows, tCols > Result;
    tData * pLeft   = ij[0];
    tData * pRight  = R.ij[0];
    tData * pResult = Result.ij[0];

    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pResult++) = (*pLeft++) + (*pRight++);

    return Result;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator - (const Matrix< tData, tRows, tCols > & R) {
    Matrix< tData, tRows, tCols >  Result;
    tData * pLeft   = ij[0];
    tData * pRight  = R.ij[0];
    tData * pResult = Result.ij[0];

    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pResult++) = (*pLeft++) - (*pRight++);
    return Result;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator * (const tData &scalar) {
    Matrix< tData, tRows, tCols >  Result = (*this);
    tData * pResult = Result.ij[0];
    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pResult++) *= scalar;
    return Result;
}

template < class tData, size_t tRows, size_t tCols > // friend
Matrix< tData, tRows, tCols > operator * (const tData & scalar,const Matrix< tData, tRows, tCols > & R) {
    size_t iRows = R.iRows;
    size_t iCols = R.iCols;
    Matrix< tData, tRows, tCols >  Result = R;
    tData * pResult = Result.ij[0];
    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pResult++) *= scalar;
    return Result;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator / (const tData &scalar) {
    assert(scalar);
    Matrix< tData, tRows, tCols >  Result = (*this);
    tData * pResult = Result.ij[0];
    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pResult++) /= scalar;
    return Result;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator *= (const Matrix< tData, tRows, tCols > & R) {
    Matrix< tData, tRows, tCols > Result;
    tData * pLeft   = ij[0];
    tData * pRight  = R.ij[0];
    tData * pResult = Result.ij[0];
    for (size_t iIndex = 0; iIndex < iRows*iCols; iIndex++)
        (*pResult++) = (*pLeft++) * tData(*pRight++);
    return Result;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator /= (const Matrix< tData, tRows, tCols > & R) {
    Matrix< tData, tRows, tCols > Result;
    tData * pLeft   = ij[0];
    tData * pRight  = R.ij[0];
    tData * pResult = Result.ij[0];
    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++) {
        assert(*pRight != 0.0);
        (*pResult++) = (*pLeft++) / tData(*pRight++);
    }
    return Result;
}

template < class tData0, size_t tCols0, size_t tRowsT, size_t tRowsB>
Matrix< tData0, tRowsT+tRowsB, tCols0 > operator & (const Matrix< tData0, tRowsT, tCols0 >& Top,
        const Matrix< tData0, tRowsB, tCols0 >& Bottom) { 
    Matrix< tData0, tRowsT+tRowsB, tCols0 > Result;

    size_t i,ii,j;
    for(i=0; i<Top.iRows; i++)
        for(j=0; j<Top.iCols; j++)
            Result[i][j] = Top[i][j];

    for(i=Top.iRows,ii=0; i<(Top.iRows+Bottom.iRows); i++,ii++) {
        for(j=0; j<Top.iCols; j++) {
            Result[i][j] = Bottom[ii][j];
        }
    }
    return Result;
}

template < class tData0, size_t tRows0, size_t tColsL, size_t tColsR >
Matrix< tData0, tRows0, tColsL+tColsR > operator | (const Matrix< tData0, tRows0, tColsL >& Left,
        const Matrix< tData0, tRows0, tColsR >& Right) { 
    Matrix< tData0, tRows0, tColsL+tColsR > Result;
    size_t i,j,jj;
    for(i=0; i<Left.iRows; i++)
        for(j=0; j<Left.iCols; j++)
            Result[i][j] = Left[i][j];
    for(i=0; i<Left.iRows; i++) {
        for(j=Left.iCols,jj=0; j<(Left.iCols+Right.iCols); j++,jj++) {
            Result[i][j] = Right[i][jj];
        }
    }
    return Result;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::zeros( void ) {
    //std::memset (ij[0], 0x0, sizeof(tData) * iRows * iCols);
    std::fill(BEGIN(storage), END(storage), tData(0));
    return *this;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::ones( void ) {
    tData * pThis = ij[0];
    for (size_t iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pThis++) = tData(1);
    return *this;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::eye( void ) {
    assert(tRows==tCols);
    //std::memset (ij[0], 0x0, sizeof(tData) * iRows * iCols);
    std::fill(BEGIN(storage), END(storage), tData(0));

    tData * pThis = ij[0];
    for (size_t iIndex = 0; iIndex < iRows; iIndex++, pThis+=iCols)
        (*pThis++) = tData(1);
    return *this;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::randn(void) {
    std::default_random_engine generator;
    std::normal_distribution<tData> distribution;
    for (size_t iIndex = 0; iIndex < iRows*iCols; iIndex++) {
        ij[0][iIndex] = distribution(generator);
    }
    return *this;
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tCols, tRows > trans( const Matrix< tData, tRows, tCols >& R ) {
    Matrix< tData, tCols, tRows > Result;

    for (size_t iIndex = 0; iIndex < tCols; iIndex++)
        for (size_t jIndex = 0; jIndex < tRows; jIndex++)
            Result[iIndex][jIndex] = R[jIndex][iIndex];

    return Result;
}

template < size_t tRows, size_t tCols >
Matrix< std::complex<float>, tCols, tRows > trans( const Matrix< std::complex<float>, tRows, tCols >& R ) {
    Matrix< std::complex<float>, tCols, tRows > Result;

    for (size_t iIndex = 0; iIndex < tCols; iIndex++)
        for (size_t jIndex = 0; jIndex < tRows; jIndex++)
            Result[iIndex][jIndex] = std::conj(R[jIndex][iIndex]);

    return Result;
}

template < size_t tRows, size_t tCols >
Matrix< std::complex<double>, tCols, tRows > trans( const Matrix< std::complex<double>, tRows, tCols >& R ) {
    Matrix< std::complex<double>, tCols, tRows > Result;

    for (size_t iIndex = 0; iIndex < tCols; iIndex++)
        for (size_t jIndex = 0; jIndex < tRows; jIndex++)
            Result[iIndex][jIndex] = std::conj(R[jIndex][iIndex]);

    return Result;
}

template < class tData, size_t tRows >
Matrix< tData, tRows, 1 > diag( const Matrix< tData, tRows, tRows >& R ) {
    Matrix< tData, tRows, 1 > Result;
    for (size_t iIndex = 0; iIndex < tRows; iIndex++ ) {
        Result[iIndex][0] = R[iIndex][iIndex];
    }
    return Result;
}

template < class tData, size_t tRows >
Matrix< tData, tRows, tRows > diag( const Matrix< tData, tRows, 1 >& R ) {
    Matrix< tData, tRows, tRows > Result;
    for (size_t iIndex = 0; iIndex < tRows; iIndex++ ) {
        Result[iIndex][iIndex] = R[iIndex][0];
    }
    return Result;
}

template < class tData, size_t tCols >
Matrix< tData, tCols, tCols > diag( const Matrix< tData, 1, tCols >& R ) {
    Matrix< tData, tCols, tCols > Result;
    for (size_t iIndex = 0; iIndex < tCols; iIndex++ ) {
        Result[iIndex][iIndex] = R[0][iIndex];
    }
    return Result;
}

template < class tData >
Matrix< tData, 3, 3 > skew( const Matrix< tData, 3, 1 >& R ) {
    tData * pR  =  R.pIJ();
    tData z = tData(0);
    Matrix< tData, 3, 3 > Result = {
        z,-pR[2], pR[1],
        pR[2],     z,-pR[0],
        -pR[1], pR[0],  z
    };

    return Result;
}

template < class tData >
Matrix< tData, 3, 1 > cross( const Matrix< tData, 3, 1 >& L, const Matrix< tData, 3, 1 >& R ) {
    return skew(L)*R;
}

template < class tData, size_t tRows >
tData dot( const Matrix< tData, tRows, 1 >& L, const Matrix< tData, tRows, 1 >& R ) {
    tData Result = tData(0);

    tData * pR = R.pIJ();
    tData * pL = L.pIJ();

    for (size_t iIndex = 0; iIndex < tRows; iIndex++) {
        Result += (*pR++)*(*pL++);
    }
    return Result;
}

template < class tData, size_t tRows, size_t tCols >
tData Matrix< tData, tRows, tCols >::n( void ) const {
    return std::sqrt(dot(*this,*this));
}

template < class tData, size_t tRows >
tData norm( const Matrix< tData, tRows, 1 >& R ) {
    return std::sqrt(dot(R,R));
}

template < class tData, size_t tRows, size_t tCols >
Matrix< tData, tRows, 1 > Matrix< tData, tRows, tCols >::u( void ) const {
    tData den = n();
    assert(den != 0.0);
    Matrix< tData, tRows, 1 > Result = *this;
    return Result*(1.0/den);
}

extern "C" void sgesv_( const int &n, const int &nrhs, float *A,
                        const int &lda, int* ipiv, float *B, const int &ldb, int *info);
extern "C" void cgesv_( const int &n, const int &nrhs, std::complex<float> *A,
                        const int &lda, int* ipiv, std::complex<float> *B, const int &ldb, int *info);
extern "C" void dgesv_( const int &n, const int &nrhs, double *A,
                        const int &lda, int* ipiv, double *B, const int &ldb, int *info);
extern "C" void zgesv_( const int &n, const int &nrhs, std::complex<double> *A,
                        const int &lda, int* ipiv, std::complex<double> *B, const int &ldb, int *info);

template < size_t tRows >
Matrix< float, tRows, tRows > inv( const Matrix< float, tRows, tRows >& R ) {
    int n = tRows;
    Matrix< float, tRows, tRows > a = R;
    int ipiv[tRows] = {0};
    Matrix< float, tRows, tRows > Result;
    Result.eye();
    int info = 0;

    sgesv_(n, n, a.pIJ(), n, ipiv, Result.pIJ(), n, &info);

    if(info != 0) {
        std::cerr << "?gesv returned error: " << info << std::endl;
        abort();
    }

    return Result;
}

template < size_t tRows >
Matrix< std::complex<float>, tRows, tRows > inv( const Matrix< std::complex<float>, tRows, tRows >& R ) {
    int n = tRows;
    Matrix< std::complex<float>, tRows, tRows > a = R;
    int ipiv[tRows] = {0};
    Matrix< std::complex<float>, tRows, tRows > Result;
    Result.eye();
    int info = 0;

    cgesv_(n, n, a.pIJ(), n, ipiv, Result.pIJ(), n, &info);

    if(info != 0) {
        std::cerr << "?gesv returned error: " << info << std::endl;
        abort();
    }

    return Result;
}

template < size_t tRows >
Matrix< double, tRows, tRows > inv( const Matrix< double, tRows, tRows >& R ) {
    int n = tRows;
    Matrix< double, tRows, tRows > a = R;
    int ipiv[tRows] = {0};
    Matrix< double, tRows, tRows > Result;
    Result.eye();
    int info = 0;

    dgesv_(n, n, a.pIJ(), n, ipiv, Result.pIJ(), n, &info);

    if(info != 0) {
        std::cerr << "?gesv returned error: " << info << std::endl;
        abort();
    }

    return Result;
}

// Specialized 2x2 and 3x3 double variants generated from sympy
// Would you like to know more? Click here: https://docs.sympy.org/latest/index.html
//   >>> from sympy import symbols, Matrix, pprint
//   >>> a,b,c,d,e,f,g,h,i = symbols('a b c d e f g h i')
//   >>> A = Matrix([[a,b,c],[d,e,f],[g,h,i]])
//   >>> pprint(A.adjugate()); pprint(A.det())
Matrix< double, static_cast<size_t>(2), static_cast<size_t>(2) >
inv( const Matrix< double, static_cast<size_t>(2), static_cast<size_t>(2) >& R ) {
    double the_det = det(R);

    if(std::abs(the_det) < FLT_EPSILON) {
        std::cerr << "matrix near singular" << std::endl;
        abort();
    }

    Matrix< double, 2, 2 > M = {R[1][1], -R[0][1], -R[1][0], R[0][0]};
    return((1.0/the_det)*M);
}

Matrix< double, static_cast<size_t>(3), static_cast<size_t>(3) >
inv( const Matrix< double, static_cast<size_t>(3), static_cast<size_t>(3) >& R ) {
    double the_det = det(R);

    if(std::abs(the_det) < FLT_EPSILON) {
        std::cerr << "matrix near singular" << std::endl;
        abort();
    }

    Matrix< double, 3, 3 > M = {R[1][1]*R[2][2] - R[1][2]*R[2][1],
                                -R[0][1]*R[2][2] + R[0][2]*R[2][1], R[0][1]*R[1][2] - R[0][2]*R[1][1],
                                -R[1][0]*R[2][2] + R[1][2]*R[2][0], R[0][0]*R[2][2] - R[0][2]*R[2][0],
                                -R[0][0]*R[1][2] + R[0][2]*R[1][0], R[1][0]*R[2][1] - R[1][1]*R[2][0],
                                -R[0][0]*R[2][1] + R[0][1]*R[2][0], R[0][0]*R[1][1] - R[0][1]*R[1][0]
                               };

    return((1.0/the_det)*M);
}

template < size_t tRows >
Matrix< std::complex<double>, tRows, tRows > inv( const Matrix< std::complex<double>, tRows, tRows >& R ) {
    int n           = tRows;
    Matrix< std::complex<double>, tRows, tRows > a = R;
    int ipiv[tRows] = {0};
    Matrix< std::complex<double>, tRows, tRows > Result;
    Result.eye();
    int info        = 0;

    zgesv_(n, n, a.pIJ(), n, ipiv, Result.pIJ(), n, &info);

    if(info != 0) {
        std::cerr << "?gesv returned error: " << info << std::endl;
        abort();
    }

    return Result;
}

extern "C" void sgetrf_(const int &m, const int &n, float *A,
                        const int &lda, int *ipiv, int *info);
extern "C" void cgetrf_(const int &m, const int &n, std::complex<float> *A,
                        const int &lda, int *ipiv, int *info);
extern "C" void dgetrf_(const int &m, const int &n, double *A,
                        const int &lda, int *ipiv, int *info);
extern "C" void zgetrf_(const int &m, const int &n, std::complex<double> *A,
                        const int &lda, int *ipiv, int *info);

template < size_t tRows >
float det( const Matrix< float, tRows, tRows >& R ) {
    int n = tRows;
    Matrix< float, tRows, tRows > a = R;
    int ipiv[tRows] = {0};
    int info = 0;
    float result = 1.0;

    sgetrf_(n,n,a.pIJ(),n,&ipiv[0],&info);

    if (info == 0) {
        for(int i=0; i<n; i++) {
            if(ipiv[i] != (i+1)) result *= -a[i][i]; // i+1 for fortran
            else                 result *= +a[i][i];
        }
    } else {
        std::cerr << "?getrf returned error: " << info << std::endl;
        abort();
    }
    return result;
}

template < size_t tRows >
std::complex<float> det( const Matrix< std::complex<float>, tRows, tRows >& R ) {
    int n = tRows;
    Matrix< std::complex<float>, tRows, tRows > a = R;
    int ipiv[tRows] = {0};
    int info = 0;
    std::complex<float> result(1.0,0.0);

    cgetrf_(n,n,a.pIJ(),n,&ipiv[0],&info);

    if (info == 0) {
        for(int i=0; i<n; i++) {
            if(ipiv[i] != (i+1)) result *= -a[i][i]; // i+1 for fortran
            else                 result *= +a[i][i];
        }
    } else {
        std::cerr << "?getrf returned error: " << info << std::endl;
        abort();
    }
    return result;
}

template < size_t tRows >
double det( const Matrix< double, tRows, tRows >& R ) {
    int n = tRows;
    Matrix< double, tRows, tRows > a = R;
    int ipiv[tRows] = {0};
    int info = 0;
    double result = 1.0;

    dgetrf_(n,n,a.pIJ(),n,&ipiv[0],&info);

    if (info == 0) {
        for(int i=0; i<n; i++) {
            if(ipiv[i] != (i+1)) result *= -a[i][i]; // i+1 for fortran
            else                 result *= +a[i][i];
        }
    } else {
        std::cerr << "?getrf returned error: " << info << std::endl;
        abort();
    }
    return result;
}

// Specialized 2x2 and 3x3 double variants generated from sympy
// Would you like to know more? Click here: https://docs.sympy.org/latest/index.html
//   >>> from sympy import symbols, Matrix, pprint
//   >>> a,b,c,d,e,f,g,h,i = symbols('a b c d e f g h i')
//   >>> A = Matrix([[a,b,c],[d,e,f],[g,h,i]])
//   >>> pprint(A.det())
double det( const Matrix< double, static_cast<size_t>(2), static_cast<size_t>(2) >& R ) {
    return( R[0][0]*R[1][1] - R[0][1]*R[1][0] );
}

double det( const Matrix< double, static_cast<size_t>(3), static_cast<size_t>(3) >& R ) {
    return( R[0][0]*R[1][1]*R[2][2] - R[0][0]*R[1][2]*R[2][1] - R[0][1]*R[1][0]*R[2][2] +
            R[0][1]*R[1][2]*R[2][0] + R[0][2]*R[1][0]*R[2][1] - R[0][2]*R[1][1]*R[2][0] );
}

template < size_t tRows >
std::complex<double> det( const Matrix< std::complex<double>, tRows, tRows >& R ) {
    int n = tRows;
    Matrix< std::complex<double>, tRows, tRows > a = R;
    int ipiv[tRows] = {0};
    int info = 0;
    std::complex<double> result(1.0,0.0);

    zgetrf_(n,n,a.pIJ(),n,&ipiv[0],&info);

    if (info == 0) {
        for(int i=0; i<n; i++) {
            if(ipiv[i] != (i+1)) result *= -a[i][i]; // i+1 for fortran
            else                 result *= +a[i][i];
        }
    } else {
        std::cerr << "?getrf returned error: " << info << std::endl;
        abort();
    }
    return result;
}

} // from namespace

#endif // from _EMATRIX_H
