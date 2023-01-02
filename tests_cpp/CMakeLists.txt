include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR})

add_executable(test_ematrix.exe test_ematrix.cpp)

target_link_libraries(test_ematrix.exe PUBLIC ${lapackblas_libraries})

