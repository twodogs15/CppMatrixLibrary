#!/usr/bin/python3

import subprocess as sp
import numpy as np
import unittest
import re
import inspect


class TestAssert(unittest.TestCase):

    dynamicStorage = 'DYNAMIC_STORAGE=1'

    def iterateFailCase(self, failCases, devNull=True):
        caller = inspect.stack()[1][3]
        base = re.compile(r'test_fail_(M\d*)').match(caller)
        caseBase = base.group(1)

        for case in failCases:
            with self.subTest(case=case):
                tag = f'FAIL_{caseBase}_{case}'
                print(f'Test Case: {tag}')
                theArgList = ['g++', '-std=c++17', '-DTESTING', f'-D{self.dynamicStorage}',
                              f'-D{tag}', '-I..', 'test_assertion_errors.cpp']
                if devNull == True:
                    child = sp.run(
                        theArgList, stdout=sp.DEVNULL, stderr=sp.STDOUT)
                else:
                    print(theArgList)
                    child = sp.run(theArgList, stderr=sp.STDOUT)
                rc = child.returncode
                self.assertEqual(rc, 0)

                child = sp.run('./a.out', capture_output=True, text=True)
                if 1 < len(failCases):
                    # This probably doesnot work.
                    ifPass = re.compile(
                        r'\[m[0-9]+\]', re.MULTILINE).search(child.stderr)
                else:
                    ifPass = re.compile(
                        f'\[{caseBase.lower()}\]', re.MULTILINE).search(child.stderr)
                self.assertIsNot(ifPass, None)

        child = sp.run(['/bin/rm', '-f', './a.out'])

    def ex_test_pass_M1(self):
        print(f'Test Case: 00')
        child = sp.run(['g++', '-std=c++17', '-DTESTING',
                        '-DPASS_M1', '-I..', 'test_compilation_errors.cpp'])
        rc = child.returncode
        self.assertEqual(rc, 0)

        child = sp.run('./a.out', capture_output=True, text=True)
        pass_00 = re.compile(r'\[m1a\]', re.MULTILINE).search(child.stderr)
        self.assertIsNot(pass_00, None)

        child = sp.run(['/bin/rm', '-f', './a.out'])

    def test_fail_M9(self):
        failCases = ['A']
        self.iterateFailCase(failCases)

    def test_fail_M10(self):
        failCases = ['A']
        self.iterateFailCase(failCases)


if __name__ == '__main__':
    unittest.main()
