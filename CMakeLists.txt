cmake_minimum_required(VERSION 3.12)

# set the project name
project(navsim VERSION 1.0 LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_SKIP_INSTALL_RULES True)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

set(BLA_VENDOR Generic)

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

add_compile_options(-Wall -Wextra -pedantic -Wno-unused-parameter 
                    -Wno-unused-variable -Wno-unused-function -fmax-errors=2)

add_subdirectory(tests_cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(readme_example.exe readme_example.cpp)
target_link_libraries(readme_example.exe PUBLIC ${lapackblas_libraries})

