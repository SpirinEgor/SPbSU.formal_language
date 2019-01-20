cmake_minimum_required(VERSION 3.10)
project(CYK_CUDA CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(CYK_CUDA main.cu reader.h)