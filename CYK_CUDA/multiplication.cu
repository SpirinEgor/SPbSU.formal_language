#include "multiplication.h"

#include <stdio.h>

// c = a * b
__global__ void mult(BLOCK_TYPE* a, BLOCK_TYPE* b, BLOCK_TYPE* c, int rows, int cols) {
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    int c_row = blockIdx.y;

    if (c_col >= cols || c_row >= rows) {
        return;
    }


    BLOCK_TYPE new_val = 0;
    for (int a_i = 0; a_i < cols; ++a_i) {
        #pragma unroll
        for (int bit = BLOCK_SIZE - 1; bit >= 0; --bit) {
            if ((a[c_row * cols + a_i] >> bit & 1)) {
                int b_row = a_i * BLOCK_SIZE + BLOCK_SIZE - 1 - bit;
                new_val |= b[b_row * cols + c_col];
            }
        }
    }
    c[c_row * cols + c_col] = new_val;
}


bool mult_with_add(BLOCK_TYPE* a, BLOCK_TYPE* b, BLOCK_TYPE* c, int rows, int cols) {
    BLOCK_TYPE* c_mult = new BLOCK_TYPE [rows * cols];

    BLOCK_TYPE* d_a;
    BLOCK_TYPE* d_b;
    BLOCK_TYPE* d_c;
    int mem_size = rows * cols * sizeof(BLOCK_TYPE);
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_a), mem_size));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_b), mem_size));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_c), mem_size));

    gpuErrchk(cudaMemcpy(d_a, a, mem_size, cudaMemcpyHostToDevice));    
    gpuErrchk(cudaMemcpy(d_b, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_c, 0, mem_size));

    dim3 blocks(cols / THREADS_NUMBER + 1, rows);
    dim3 threads(THREADS_NUMBER);
    mult<<<blocks, threads>>>(d_a, d_b, d_c, rows, cols);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(c_mult, d_c, mem_size, cudaMemcpyDeviceToHost));

    bool has_changed = false;
    for (int _r = 0; _r < rows; ++_r) {
        for (int _c = 0; _c < cols; ++_c) {
            if (c[_r * cols + _c] != (c[_r * cols + _c] | c_mult[_r * cols + _c])) {
                has_changed = true;
                c[_r * cols + _c] |= c_mult[_r * cols + _c];
            }
        }
    }
    return has_changed;
}
