#!/bin/bash

INPUT_CHOMSKY_FILE=$1
INPUT_GRAPH_FILE=$2
OUTPUT_FILE=$3

# call your programm here
g++ --std=c++17 simple_solution.cpp
./a.out $1 $2 $3
