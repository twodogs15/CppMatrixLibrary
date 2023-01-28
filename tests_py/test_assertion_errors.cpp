/*! This file is part of EMatrix, the C++ matrix library distribution.
 *  This project is licensed under the terms of the MIT license. The full text
 *  of the license file can be found in LICENSE.
 */

/// \file

#include <iostream>
#include <climits>
#include <csignal>

#include "EMatrix.h"

using namespace ematrix;
using namespace std;

#ifndef THE_TYPE
#define THE_TYPE double
#endif

void segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
    printf("Caught segfault at address %p\n", si->si_addr);
    exit(0);
}


int main(void) {

/// [m0a] Matrix memory allocation/storage assignment.
#ifdef FAIL_M0_A
    struct sigaction sa;

    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = segfault_sigaction;
    sa.sa_flags   = SA_SIGINFO;

    sigaction(SIGSEGV, &sa, NULL);

try {
        Matrix< THE_TYPE, 1, (1ul<<28) > A;
} catch (std::exception& e) {
    std::cerr << "Exception caught : " << e.what() << std::endl;
}
 #endif

/// [m9] STL list initialize constructor.
/// Assertion fail for too many elements.
#ifdef FAIL_M9_A
    Matrix< THE_TYPE, 2, 3 > A = {1,2,3,4,5,6,7};
#endif

/// [m10] STL list initialize constructor.
/// Assertion fail for too many elements.
#ifdef FAIL_M10_A
    Matrix< THE_TYPE, 2, 3 > A;    
    A = {1,2,3,4,5,6,7};
#endif
    
    return(0);
}

