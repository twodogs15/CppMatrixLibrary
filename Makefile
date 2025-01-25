# This file is part of EMatrix, the C++ matrix library distribution.
# This project is licensed under the terms of the MIT license. The full text
# of the license file can be found in LICENSE.
 
TGT1      = readme_example.exe                    
TGT2      = test_ematrix.exe

SRC1      = readme_example.cpp
SRC2      = ./tests_cpp/test_ematrix.cpp

OBJ1     := $(SRC1:.cpp=.o)
OBJ2     := $(SRC2:.cpp=.o)

CXX      = g++ # clang++
CC       = gcc # clang
INCLUDES = -I. 
CXXFLAGS = $(INCLUDES) -std=c++17 -pedantic -Wall -Wextra -O2 -Wno-unused-variable -Wfatal-errors# -DDYNAMIC_STORAGE

LDFLAGS  =# -llapack -lblas 

all : $(TGT1) $(TGT2)

$(TGT1) : $(OBJ1)
	$(CXX) -o $(TGT1) $(OBJ1) $(LDFLAGS)
$(TGT2) : $(OBJ2)
	$(CXX) -o $(TGT2) $(OBJ2) $(LDFLAGS)

$(OBJ1): $(SRC1) EMatrix.h
$(OBJ2): $(SRC2) EMatrix.h

clean :
	@-rm -f $(OBJ1) $(TGT1) $(OBJ2) $(TGT2)

