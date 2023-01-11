cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_SKIP_INSTALL_RULES True)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

find_package(PythonInterp REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/..)

#add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)

set(HEADERS EMatrix.h)

pybind11_add_module(test_EMatrix test_EMatrix.cpp) 
set_target_properties(test_EMatrix PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(test_ctors test_ctors.cpp) 
set_target_properties(test_ctors PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
