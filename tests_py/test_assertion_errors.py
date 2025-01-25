#!/usr/bin/python3

import subprocess as sp
import numpy as np
import unittest
import re
import inspect


class TestAssert(unittest.TestCase):

    dynamicStorage = 'DYNAMIC_STORAGE=0'

    def iterateFailCase(self, failCases, devNull=False):
        caller = inspect.stack()[1][3]
        base = re.compile(r'test_fail_(M\d*)').match(caller)
        caseBase = base.group(1)

        for case in failCases:
            with self.subTest(case=case):
                tag = f'FAIL_{caseBase}_{case}'
                print(f'Test Case: {tag}')
				# sudo apt-get install libc++-dev
                theArgList = ['clang++', '-std=c++17', '-stdlib=libc++', '-DTESTING', f'-D{self.dynamicStorage}',
                              f'-D{tag}', '-I..', 'test_assertion_errors.cpp']
                if devNull == True:
                    child = sp.run(
                        theArgList, stdout=sp.DEVNULL, stderr=sp.STDOUT)
                else:
                    print(' '.join(theArgList))
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
                        f'[{caseBase.lower()}]', re.MULTILINE).search(child.stderr)

                self.assertIsNot(ifPass, None)

        #child = sp.run(['/bin/rm', '-f', './a.out'])

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

    # M0 is the allocation function.  In the case of dynamic storage we use new.
    # The static allocation is a local variable.  This function is difficult to
    # test as the assertion fail conditions are system dependent.  The default
    # Linux stack limit is 8 MB.  On a Chromebook Linux virtual machine, The
    # max dynamic object Matrix< double, 1, (1ul<<27) > has an object size of
    # 24 bytes, but allocates 1 GB of dynamic memory.  A matrix of
    # 1 x (1ul<<28) fails. In the static case, the object
    # Matrix< double, 1, (1ul<<20) - 0x0427 > constructs with an object size of
    # almost 8 MB.  So that we have something to test.  Also, these errors seg
    # fault and cannot be caught by exception and does not cause a compilation
    # error so we are just going to turn this test off.

    def x_test_fail_M0(self):
        failCases = ['A']
        self.iterateFailCase(failCases)

    def test_fail_M9(self):
        failCases = ['A']
        self.iterateFailCase(failCases)

    def test_fail_M10(self):
        failCases = ['A']
        self.iterateFailCase(failCases)


if __name__ == '__main__':
    unittest.main()
