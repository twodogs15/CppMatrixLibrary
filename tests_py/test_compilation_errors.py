#!/usr/bin/python3

import subprocess as sp
import numpy as np
import unittest
import re
import inspect


class TestCompilation(unittest.TestCase):

    dynamicStorage = 'DYNAMIC_STORAGE=0'

    def iterateFailCase(self, failCases, devNull=True):
        caller = inspect.stack()[1][3]
        base = re.compile(r'test_fail_(M\d*)').match(caller)
        caseBase = base.group(1)

        for case in failCases:
            with self.subTest(case=case):
                tag = f'FAIL_{caseBase}_{case}'
                print(f'Test Case: {tag}')
                theArgList = ['g++', '-std=c++17', '-fsyntax-only', '-DTESTING', f'-D{self.dynamicStorage}',
                              f'-D{tag}', '-I..', 'test_compilation_errors.cpp']
                if devNull == True:
                    child = sp.run(
                        theArgList, stdout=sp.DEVNULL, stderr=sp.STDOUT)
                else:
                    print(' '.join(theArgList))
                    child = sp.run(
                        theArgList, stderr=sp.STDOUT)
                rc = child.returncode
                self.assertNotEqual(rc, 0)


    def stub_test_fail_M0(self):
        failCases = ['A']
        self.iterateFailCase(failCases)

    def test_pass_M1(self):
        child = sp.run(['g++', '-std=c++17', '-DTESTING', f'-D{self.dynamicStorage}',
                        '-DPASS_M1', '-I..', 'test_compilation_errors.cpp'])
        rc = child.returncode
        self.assertEqual(rc, 0)

        child = sp.run('./a.out', capture_output=True, text=True)
        pass_00 = re.compile(r'\[m1[ab]\]', re.MULTILINE).search(child.stderr)
        self.assertIsNot(pass_00, None)

        child = sp.run(['/bin/rm', '-f', './a.out'])

    def test_fail_M2(self):
        failCases = ['A', 'B', 'C', 'D', 'E']
        self.iterateFailCase(failCases)

    def test_fail_M3(self):
        failCases = ['A', 'B', 'C']
        self.iterateFailCase(failCases)

    def test_fail_M4(self):
        failCases = ['A']
        self.iterateFailCase(failCases)


if __name__ == '__main__':
    unittest.main()
