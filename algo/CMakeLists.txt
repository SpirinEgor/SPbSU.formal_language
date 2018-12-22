cmake_minimum_required(VERSION 3.12)
project(AlgoWithM4ri)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -lm4ri)

add_executable(AlgoWithM4ri main.cpp graph.cpp graph.h grammar.cpp grammar.h)