cmake_minimum_required(VERSION 3.5)
project(CYK LANGUAGES CXX CUDA)

add_executable(CYK cuda_solution.cu)
set_property(TARGET CYK PROPERTY CUDA_SEPARABLE_COMPILATION ON)