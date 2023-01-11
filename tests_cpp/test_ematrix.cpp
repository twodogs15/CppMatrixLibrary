/*! This file is part of EMatrix, the C++ matrix library distribution.
 *  This project is licensed under the terms of the MIT license. The full text
 *  of the license file can be found in LICENSE.                         
 */

/// \file

#include <iostream>

#include "EMatrix.h"

using namespace ematrix;
using namespace std;

int main(void) {
//! Begin and the very beginning.

    // Scoped to test [virtual ~Matrix ()], used gdb and cerr to verify
    {
        // Matrix ();
        // void matalloc (size_t iRowIndex, size_t iColIndex);
        Matrix<double,2,3> A;

        // These invoke compiler errors
        //Matrix<double,0,0> A;
        //Matrix<double,0,1> A;
        //Matrix<double,1,0> A;
    }

    {
        // inline Matrix (const Matrix< tData, tRows, tCols > & R);
        // std::ostream& operator << (std::ostream& s,const Matrix< tData, tRows, tCols >& A)
        float a[2][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0}};
        Matrix<float,2,3> A(&a[0][0]);
        Matrix<float,2,3> B=A;
        B = B+B+A;
        // Matrix<float,3,2> B=A; // Compiler error
        cerr << B+B+A << endl;
    }

    {
        // Matrix (tData* tArray);
        float a[2][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0}};
        Matrix<float,2,3> A(&a[0][0]);
        cerr << A << endl;;
    }

    {
        // Matrix (const std::initializer_list<tData>& l);
        Matrix<double,3,3> A = {1.,2.,3.,0.,0.,0.,0,0,0};
        cerr << A << endl;;

        // Assertion fail, too many elements
        //Matrix<double,3,3> B = {1.,2.,3.,0.,0.,0.,0,0,0,0};
        // Assertion fail, too few elements
        //Matrix<double,3,3> B = {1.,2.,3.,0.,0.,0.,0,0};
    }

    /* Deprecated but tested
    { // Matrix (const std::initializer_list<tData>& l);
        Matrix<double, 1, 2 > X(2.81,3.14);
        cerr << X;
        Matrix<double, 1, 3 > Y(2.81,3.14,FLT_MAX);
        cerr << Y;
    }
    */

    {
        // inline Matrix (const Matrix< tData, tRows, tCols > & R);
        Matrix<double,3,3> A = {1.2,2.2,3.2,4.,5.,0.,0,0,0};
        Matrix<double,3,3> B;
        Matrix<double,2,2> C;
        B=A;
        cerr << B << endl;
        //B=B;
        cerr << B << endl;
        // B=C; // Compiler error
    }

    {
        // inline const Matrix< tData, tRows, tCols > &operator = (const std::initializer_list<tData>& l);
        Matrix<double,3,2> A;
        A = {1.1,2.1,3.1,0.0,0.0,0.0};
        cerr << A << endl;
        // Assertion fail, too many elements
        Matrix<double,3,3> B;
        //B = {1.,2.,3.,0.,0.,0.,0,0,0,0};
        // Assertion fail, too few elements
        //B = {1.,2.,3.,0.,0.,0.,0,0};
    }

    {
        // load(tData* tArray)
        double a[2][3] = {{1.1,2.1,3.1},{4.0,5.0,6.0}};
        Matrix<double,3,2> A;
        A.load(&a[0][0]);
        cerr << A << endl;
    }

    {
        // inline tData * operator [] (size_t iRowIndex) const
        // inline tData & operator () (size_t iRowIndex, size_t iColIndex) const;
        // inline tData & operator () (size_t iIndex) const;
        Matrix<double,3,2> A = {1,2,3,4,5,6};
        A[0][0] = 7;
        cerr << A[2][1] << endl;
        cerr << A << endl;
        // A[10][0] = 7; // assertion fail,
        // cerr << A[11][2] << endl;
        // cerr << A[1][12] << endl; // Not safe for col.

        A(1,1) = 8;
        cerr << A(3,2) << endl;
        cerr << A << endl;

        //cerr << A(1,12) << endl; // assertion fail,
        //cerr << A(12,1) << endl;
        //cerr << A(0,1) << endl;
        //cerr << A(1,0) << endl;
        //A(1,12) = 3.14;
        //A(12,1) = 3.14;
        //A(0, 1)  = 3.14;
        //A(1,-1)  = 3.14;

        Matrix<double,6,1> V = {1,2,3,4,5,6};
        V(1) = 8;
        cerr << V << endl;
        cerr << V(6) << endl;

        //V(12) = 3.14;// assertion fail,
        //cerr << V(12) << endl;
        //V(0) = 3.14;
        //V << V(0) << endl;

        Matrix<double,1,6> U = {1,2,3,4,5,6};
        U(1) = 8;
        cerr << U << endl;
        cerr << U(6) << endl;

        //U(12) = 3.14;// assertion fail,
        //cerr << U(12) << endl;
        //U(0) = 3.14;
        //cerr << U(0) << endl;
    }

    /* Deprecated but tested
    {  // inline ListInit<tData, tData* > operator = (tData x) {
        Matrix<double,1,3> A;
        A = 1.1,2.1;
        cerr << A;
        A = 1.1,2.1,3.1;
        cerr << A;
        A = 1.1,2.1,3.1;
        cerr << A;
    */

    {
        double a[2][3] = {{1.40,2.40,3.40},{4.40,5.40,6.40}};
        Matrix<double,2,3> A(&a[0][0]);

        cerr << A << endl;

        Matrix< complex<double>,3,3 > ZA;
        ZA[0][0] = complex<double>( 3.1,-1.8);
        ZA[0][1] = complex<double>( 1.3, 0.2);
        ZA[0][2] = complex<double>(-5.7,-4.3);
        ZA[1][0] = complex<double>( 1.0, 0.0);
        ZA[1][1] = complex<double>(-6.9, 3.2);
        ZA[1][2] = complex<double>( 5.8, 2.2);
        ZA[2][0] = complex<double>( 3.4,-4.0);
        ZA[2][1] = complex<double>( 7.2, 2.9);
        ZA[2][2] = complex<double>(-8.8, 3.2);

        //Matrix< complex<double>,3,3 > ZAinv = inv(ZA);
        //octave(cout,ZA,"za") << endl;
        //cerr << trans(ZA) << endl;
        //cerr << ZAinv << endl;
        //cerr << det(ZA) << endl;
        //cerr << det(trans(ZA)) << endl;
    }

    {
        // std::ostream& operator << (std::ostream& s,const Matrix< tData, tRows, tCols >& A)
        // inline tData *pIJ (void) const {
        float a[2][3] = {{1.40f,2.40f,3.40f},{4.40f,5.40f,6.40f}};
        Matrix<float,2,3> A(&a[0][0]);
        Matrix<float,2,3> B=A;
        // Matrix<float,3,2> B=A; // Compiler error
        cerr << "p " << cout.precision(10) << endl;
        cerr << B << endl;

        float* ptr = A.pIJ();
        cerr << ptr[4] << endl;
        cerr << "rows = " << A.rows() << endl;
        cerr << "cols = " << A.cols() << endl;
    }

    {
        // bool operator == (Matrix< tData, tRows, tCols > & R);
        // bool operator != (Matrix< tData, tRows, tCols > & R);
        // Matrix< tData, tRows, tCols > operator + ();
        // Matrix< tData, tRows, tCols > operator - ();
        // Matrix< tData, tRows, tCols > operator + (const Matrix< tData, tRows, tCols > & R)
        // Matrix< tData, tRows, tCols > operator - (const Matrix< tData, tRows, tCols > & R)
        Matrix<int,2,2> A = {1,2,3,4};
        Matrix<int,2,2> B = {1,2,3,4};

        if( A == B ) cerr << "( A == B )" << endl;
        else cerr << "( A != B )" << endl;
        B(2,2) = 1;
        if( A == B ) cerr << "( A == B )" << endl;
        else cerr << "( A != B )" << endl;

        B(2,2) = 4;
        if( A != B ) cerr << "( A != B )" << endl;
        else cerr << "( A == B )" << endl;
        B(2,2) = 1;
        if( A != B ) cerr << "( A != B )" << endl;
        else cerr << "( A == B )" << endl;

        Matrix<float,2,2> C = {1,2,3,4};
        Matrix<float,2,2> D = (+C);

        if (C == D) cerr << "(C == D)" << endl;
        else cerr << "(C != D)" << endl;

        cerr << -C + D << endl;
        cerr <<  C - D << endl;

    }

    {
        // Matrix< tData, tRows, tCols > operator * (const tData & scalar);
        // friend Matrix< tData, tRows, tCols > operator * (const tData & scalar,const Matrix< tData, tRows, tCols > & R);
        // Matrix< tData, tRows, tCols > operator / (const tData & scalar);
        // Matrix< tData, tRows, tColsR > operator * (const Matrix< tData, tCols, tColsR >& R)

        Matrix<double,3,2> A = {1,2,3,4,5,6};
        cerr << A*(-1.0) << endl;
        cerr << -2.0*A << endl;
        cerr << A/2.0 << endl;
        // cerr << A/0.0 << endl; // assertion error

        Matrix<double,2,3> B = {1,2,3,4,5,6};
        Matrix<double,2,3> C = {1,2,3,4,5,6};

        cerr << A*B << endl;
        cerr << B*A << endl;
        // cerr << B*C << endl; // compiler error
    }

    {
        // Matrix< tData, tRows, tCols > operator *= (const Matrix< tData, tRows, tCols >  & R);
        // Matrix< tData, tRows, tCols > operator /= (const Matrix< tData, tRows, tCols >  & R);

        Matrix<double,2,3> B = {1,2,3,4,5,6};
        Matrix<double,2,3> C = {1,2,3,4,5,6};
        Matrix<double,3,2> D = {1,2,3,4,5,6};
        cerr << (B*=C) << endl;
        cerr << (B/=C) << endl;
        //C(1,1) = 0;
        //cout << (B/=C) << endl; // Divide by zero error.
        //cout << (B*=D) << endl; // Compiler error
        //cout << (B/=D) << endl; // Compiler error

    }

    {
        // friend Matrix< tData0, tRowsT+tRowsB, tCols0 > operator & (const Matrix< tData0, tRowsT, tCols0 >& Top,
        //         const Matrix< tData0, tRowsB, tCols0 >& Bottom);
        // friend Matrix< tData0, tRows0, tColsL+tColsR > operator | (const Matrix< tData0, tRows0, tColsL >& Left,
        //         const Matrix< tData0, tRows0, tColsR >& Right);

        Matrix<double,2,3> B = {1,2,3,4,5,6};
        Matrix<double,2,3> C = {1,2,3,4,5,6};

        Matrix<double,4,3> D = (B&C);
        Matrix<double,2,2> E = {1,2,3,4};

        cerr << (B&C) << endl;
        cerr << (B|C) << endl;
        cerr << (D & D) << endl;

        // D|B; compiler
        // B|D; compiler
        // B&E; compiler
        // E&B; compiler
    }

    {
        // Matrix< tData, tRows, tCols > zeros( void );
        // Matrix< tData, tRows, tCols > ones( void );
        // Matrix< tData, tRows, tCols > eye( void );
        // Matrix< tData, tRows, tCols > randn(void);

        Matrix<double,2,2> A = {1,2,3,4};
        cerr << A << endl;
        cerr << A.eye() << endl;

        // Matrix<double,3,2> B;
        // cout << B.eye() << endl; // assertion error

        cerr << A.zeros() << endl;
        cerr << A.ones() << endl;
        Matrix< float, 100,1 > n;
        cerr << n.randn() << endl;
    }

    {
        Matrix<double,2,3> B = {1,2,3,4,5,6};

        cerr << B << endl;
        cerr << trans(B) << endl;

        Matrix< complex<double>,3,2> ZA;
        ZA[0][0] = complex<double>( 3.1,-1.8);
        ZA[0][1] = complex<double>( 1.3, 0.2);
        ZA[0][2] = complex<double>(-5.7,-4.3);
        ZA[1][0] = complex<double>( 1.0, 0.0);
        ZA[1][1] = complex<double>(-6.9, 3.2);
        ZA[1][2] = complex<double>( 5.8, 2.2);
        cerr << ZA << endl;
        cerr << trans(ZA) << endl;

        Matrix< complex<float>,3,2> ZB;
        ZB[0][0] = complex<float>( 3.1f,-1.8f);
        ZB[0][1] = complex<float>( 1.3f, 0.2f);
        ZB[0][2] = complex<float>(-5.7f,-4.3f);
        ZB[1][0] = complex<float>( 1.0f, 0.0f);
        ZB[1][1] = complex<float>(-6.9f, 3.2f);
        ZB[1][2] = complex<float>( 5.8f, 2.2f);
        cout << ZB << endl;
        cout << trans(ZB) << endl;

    }

    {
        // friend Matrix< tData0, tRows0, 1 > diag( const Matrix< tData0, tRows0, tRows0 >& R ); // tested
        // friend Matrix< tData0, tRows0, tRows0 > diag( const Matrix< tData0, tRows0, 1 >& R ); // tested
        // friend Matrix< tData0, tCols0, tCols0 > diag( const Matrix< tData0, 1, tCols0 >& R ); // tested

        Matrix<double,1,6> A = {1,2,3,4,5,6};
        Matrix<double,6,1> B = {1,2,3,4,5,6};
        Matrix<double,2,2> C = {1,2,3,4};
        cerr << (diag(C)) << endl;

        cerr << diag(A) << endl;
        cerr << diag(B) << endl;
    }

    {
        // friend Matrix< tData0, 3, 3 > skew( const Matrix< tData0, 3, 1 >& R );
        // friend Matrix< tData0, 3, 1 > cross( const Matrix< tData0, 3, 1 >& L, const Matrix< tData0, 3, 1 >& R );
        Matrix<double,3,1> B = {2,3,4};
        Matrix<double,3,1> C = {1,2,3};
        cerr << skew(C) << endl;

        Matrix<double,3,1> x = {1,0,0};
        Matrix<double,3,1> y = {0,1,0};
        cerr << cross(x,y) << endl;

        cerr << dot(B,C) << endl;

        cerr << B.n() << endl;
        cerr << norm(B) << endl;

        cerr << B.u() << endl;
        cerr << norm(B.u()) << endl;

        //Matrix<double,2,3> A = {1,2,3,4,5,6};
        //cerr << A.n() << endl;
    }

    {
        double a[2][3] = {{1.40,2.40,3.40},{4.40,5.40,6.40}};
        Matrix<double,2,3> A(&a[0][0]);

        cerr << A << endl;

        Matrix< complex<double>,3,3 > ZA;
        ZA[0][0] = complex<double>( 3.1,-1.8);
        ZA[0][1] = complex<double>( 1.3, 0.2);
        ZA[0][2] = complex<double>(-5.7,-4.3);
        ZA[1][0] = complex<double>( 1.0, 0.0);
        ZA[1][1] = complex<double>(-6.9, 3.2);
        ZA[1][2] = complex<double>( 5.8, 2.2);
        ZA[2][0] = complex<double>( 3.4,-4.0);
        ZA[2][1] = complex<double>( 7.2, 2.9);
        ZA[2][2] = complex<double>(-8.8, 3.2);

        //Matrix< complex<double>,3,3 > ZAinv = inv(ZA);

        //cerr << det(ZA) << endl;
        //cerr << det(trans(ZA)) << endl;
    }

    {
        float a[2][3] = {{1.40f,2.40f,3.40f},{4.40f,5.40f,6.40f}};
        Matrix<float,2,3> A(&a[0][0]);

        cerr << A << endl;

        Matrix< complex<float>,3,3 > ZC;
        ZC[0][0] = complex<float>( 3.1f,-1.8f);
        ZC[0][1] = complex<float>( 1.3f, 0.2f);
        ZC[0][2] = complex<float>(-5.7f,-4.3f);
        ZC[1][0] = complex<float>( 1.0f, 0.0f);
        ZC[1][1] = complex<float>(-6.9f, 3.2f);
        ZC[1][2] = complex<float>( 5.8f, 2.2f);
        ZC[2][0] = complex<float>( 3.4f,-4.0f);
        ZC[2][1] = complex<float>( 7.2f, 2.9f);
        ZC[2][2] = complex<float>(-8.8f, 3.2f);

        //Matrix< complex<float>,3,3 > ZCinv = inv(ZC);

        //cerr << det(ZC) << endl;
        //cerr << det(trans(ZC)) << endl;
    }

    {
        Matrix< double, 3, 3 > A = {1, 1, 1, 2, 3, 4, 1, 3, 6};
        Matrix< double, 3, 3 > B = {6, -3, 1, -8, 5, -2, 3, -2, 1};
        cerr << inv(A) << endl;
        cerr << inv(B) << endl;
        cerr << det(A) << endl;
        //Matrix< double, 4, 4 > C; C.randn();
        //cerr << det(C) << endl;
    }

    {
        Matrix< float, 3, 3 > A = {1, 1, 1, 2, 3, 4, 1, 3, 6};
        Matrix< float, 3, 3 > B = {6, -3, 1, -8, 5, -2, 3, -2, 1};
        //cerr << inv(A) << endl;
        //cerr << inv(B) << endl;
        //cerr << det(A) << endl;

        //float a[2][3] = {{1.40,2.40,3.40},{4.40,5.40,6.40}};
        //Matrix<float,2,3> C(&a[0][0]);
        //cerr << det(C) << endl; // assertion error
    }

    cerr << "Done!" << endl;

    return(0);
}
