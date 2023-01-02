#!/usr/bin/python3


import subprocess as sp
import numpy as np 
import unittest

import sys
sys.path.insert(0, './build/glue')

class TestSum(unittest.TestCase):
    def test_pass_00(self):
        child = sp.run(['g++', '-std=c++17', '-I..', '-fsyntax-only', 'test_compilation_errors.cpp'])
        rc = child.returncode
        self.assertEqual(rc,0)

    def test_fail_01(self):
        child = sp.run(['g++', '-std=c++17', '-DFAIL_01', '-I..', '-fsyntax-only', 'test_compilation_errors.cpp'], 
                stdout=sp.DEVNULL, stderr=sp.STDOUT)
        rc = child.returncode
        self.assertNotEqual(rc,0)

    def test_fail_02(self):
        child = sp.run(['g++', '-std=c++17', '-DFAIL_02', '-I..', '-fsyntax-only', 'test_compilation_errors.cpp'], 
                stdout=sp.DEVNULL, stderr=sp.STDOUT)
        rc = child.returncode
        self.assertNotEqual(rc,0)

    def test_fail_03(self):
        child = sp.run(['g++', '-std=c++17', '-DFAIL_03', '-I..', '-fsyntax-only', 'test_compilation_errors.cpp'], 
                stdout=sp.DEVNULL, stderr=sp.STDOUT)
        rc = child.returncode
        self.assertNotEqual(rc,0)

if __name__ == '__main__':
    unittest.main() 

