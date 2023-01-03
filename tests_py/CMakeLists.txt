cmake_minimum_required(VERSION 3.12)

project(test VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_SKIP_INSTALL_RULES True)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

find_package(PythonInterp REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

#add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)

project(code CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(HEADERS
    EMatrix.h)

set(SOURCES test_EMatrix.cpp) 

pybind11_add_module(code ${SOURCES}) 

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
pybind11_add_module(test_ctors test_ctors.cpp) 
