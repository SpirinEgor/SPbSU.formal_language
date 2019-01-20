#ifndef MULT
#define MULT

#include <iostream>

#include "params.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl; 
        if (abort) {
            exit(code);
        }
    }
}

bool mult_with_add(BLOCK_TYPE* a, BLOCK_TYPE* b, BLOCK_TYPE* c, int rows, int cols);

__global__ void mult(BLOCK_TYPE* a, BLOCK_TYPE* b, BLOCK_TYPE* c, int rows, int cols);

#endif  // MULT