include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR})

set(BLA_VENDOR Generic)

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif() 

add_executable(test_ematrix.exe test_ematrix.cpp)

target_link_libraries(test_ematrix.exe PUBLIC ${lapackblas_libraries})

