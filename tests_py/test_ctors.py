#!/usr/bin/python3

import numpy as np 
import unittest

import sys
sys.path.insert(0, '../build/tests_py')

import test_ctors

class testCtors(unittest.TestCase):

    def test_ctor_m1(self):
        a = test_ctors.test_ctor_m1()
        b = np.array(a)
        c = np.array([[0,0,0], [0,0,0]]).astype(np.float64)
        
        self.assertEqual(b.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(b,c))

    def test_ctor_m2(self):
        a = np.array([[1,2,3],[4,5,6]])
        b = test_ctors.test_ctor_m2(a)
        c = np.array(b)
        
        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a,c))
        
if __name__ == '__main__':    
    unittest.main()

