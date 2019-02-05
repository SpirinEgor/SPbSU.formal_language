#!/bin/bash

echo "----- Install dependencies -----"
# simple solution
g++ --std=c++17 -O3 solutions/simple_solution.cpp -o main
# CUDA solution
# ./solutions/cuda_solution/build.sh
# cp solutions/cuda_solution/build/main main
