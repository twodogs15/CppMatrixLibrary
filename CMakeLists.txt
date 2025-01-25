cmake_minimum_required(VERSION 3.27)

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

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror
                        -Wno-unused-variable -Wno-unused-function -fmax-errors=2)    
endif()

add_compile_definitions(DYNAMIC_STORAGE=0)
add_compile_definitions(TESTING)

add_subdirectory(tests_cpp)

find_package(pybind11)
if(pybind11_FOUND)
    add_subdirectory(tests_py)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(readme_example.exe readme_example.cpp)
target_link_libraries(readme_example.exe PUBLIC ${lapackblas_libraries})

