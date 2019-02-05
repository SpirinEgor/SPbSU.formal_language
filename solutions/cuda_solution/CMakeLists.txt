cmake_minimum_required(VERSION 3.10)
project(CFPQ_CUDA CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(main main.cu multiplication.cu multiplication.h reader.h params.h)
add_executable(test_mult test.cu multiplication.cu multiplication.h params.h)