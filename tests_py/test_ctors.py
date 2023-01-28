#!/usr/bin/python3

import test_ctors
import numpy as np
import unittest
import inspect

class testCtors(unittest.TestCase):

    verbose = False

    def printStatus(self, caller):
        if self.verbose is True:
            print(f'{caller} finished\n')
        
    # [m2] Default constructor.
    def test_ctor_m2(self):
        a = test_ctors.test_ctor_m2()
        b = np.array(a)
        c = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float64)

        self.assertEqual(b.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(b, c))
        self.printStatus(inspect.stack()[0][3])

    # [m3] Copy constructor.
    def test_ctor_m3(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = test_ctors.test_ctor_m3(a)
        c = np.array(b)

        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a, c))
        self.printStatus(inspect.stack()[0][3])
 
    # [m4] Copy assignment operator.
    def test_ctor_m4(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = test_ctors.test_ctor_m4(a)
        c = np.array(b)

        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a, c)) 
        self.printStatus(inspect.stack()[0][3])

    #  No test for m5
  
    # [m6b] Move constructor.
    def test_ctor_m6b(self):
        if self.verbose is True:
            print('Calls [m3] Copy constructor when DYNAMIC_STORAGE == 0')

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = test_ctors.test_ctor_m6b(a)
        c = np.array(b)

        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a, c))  
        self.printStatus(inspect.stack()[0][3])

    # [m7b] Move assignment operator.
    def test_ctor_m7b(self):
        if self.verbose is True:
            print('Calls [m4] Copy assignment operator when DYNAMIC_STORAGE == 0')

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = test_ctors.test_ctor_m7b(a)
        c = np.array(b)

        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a+a, c))  
        self.printStatus(inspect.stack()[0][3])

    # [m8] Memory, i.e. pointer or array, initialize constructor.
    def test_ctor_m8(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = test_ctors.test_ctor_m8(a)
        c = np.array(b)

        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a, c))  
        self.printStatus(inspect.stack()[0][3])

    # [m9] STL list initialize constructor.
    def test_ctor_m9(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = test_ctors.test_ctor_m9(a)
        c = np.array(b)

        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a, c))  
        self.printStatus(inspect.stack()[0][3])

    # [m10] STL list initialize constructor.
    def test_ctor_m10(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = test_ctors.test_ctor_m10(a)
        c = np.array(b)

        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a, c))  
        self.printStatus(inspect.stack()[0][3])

    # [m11] Builtin C/C++ array copy to Matrix.
    def test_ctor_m11(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = test_ctors.test_ctor_m11(a)
        c = np.array(b)

        self.assertEqual(a.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(a, c))  
        self.printStatus(inspect.stack()[0][3])


    # [m16] Submatrix assignment (n-1) based.
    def test_ctor_m16(self):
        a = np.array([[1, 2], [4, 5]])
        b = test_ctors.test_ctor_m16(a)
        c = np.array(b)
        d = np.array([[0, 1, 2], [0, 4, 5]])

        self.assertEqual(d.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(d, c))  
        self.printStatus(inspect.stack()[0][3])

    # [m17] Submatrix extraction (n-1) based.
    def test_ctor_m17(self):
        a = np.array([[0, 0, 1, 2, 3, 0],
                      [0, 0, 4, 5, 6, 0],
                      [0, 0, 0, 0, 0, 0]])
        b = test_ctors.test_ctor_m17(a)
        c = np.array(b)
        d = np.array([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(d.shape, c.shape)
        self.assertIsNone(np.testing.assert_array_equal(d, c))  
        self.printStatus(inspect.stack()[0][3])

if __name__ == '__main__':
    unittest.main()
