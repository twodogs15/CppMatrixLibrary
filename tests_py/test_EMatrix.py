#!/usr/bin/python3


import numpy as np 
import unittest

import sys
sys.path.insert(0, '../build/tests_py')

import code

class TestSum(unittest.TestCase):

    def test_sum(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
        m2 = np.array([[10, 20, 30], [40, 50, 60]]).astype(np.float64)
        m3 = code.myMatrix(m1)
        m4 = code.myMatrix(m2)
        m5 = np.array( code.add(m3,m4) )
        self.assertIsNone(np.testing.assert_array_equal(m5, m1+m2))

    def test_diff(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
        m2 = np.array([[10, 20, 30], [40, 50, 60]]).astype(np.float)
        m3 = code.myMatrix(m1)
        m4 = code.myMatrix(m2)
        m5 = np.array( code.sub(m3,m4) )
        self.assertIsNone(np.testing.assert_array_equal(m5, m1-m2))

    def test_neg(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
        m3 = code.myMatrix(m1)
        m5 = np.array( code.neg(m3) )
        self.assertIsNone(np.testing.assert_array_equal(m5, -m1))

    def test_scalar_mult(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
        m3 = code.myMatrix(m1)
        m5 = np.array( code.scalar_mult(-1, m3) )
        self.assertIsNone(np.testing.assert_array_equal(m5, -m1))

    def test_mult_scalar(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
        m3 = code.myMatrix(m1)
        m5 = np.array( code.mult_scalar(m3, -1) )
        self.assertIsNone(np.testing.assert_array_equal(m5, -m1))

if __name__ == '__main__':    
    m1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
    a = code.f(m1)
    print( np.array(a) )
    unittest.main()

