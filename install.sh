#!/bin/bash

echo "----- Install dependencies -----"
# simple solution
g++ --std=c++17 -O3 solutions/simple_solution.cpp -o main
# CUDA solution
# cd solutions/cuda_solution
# ./build.sh
# cd ../..
# cp solutions/cuda_solution/build/main main
