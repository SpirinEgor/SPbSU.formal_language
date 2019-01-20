#include "multiplication.h"

#include <stdio.h>

__device__ bool _has_changed;
__global__ void _mult_with_add(
    BLOCK_TYPE* a, BLOCK_TYPE* b, BLOCK_TYPE* c, int rows, int cols) {
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    int c_row = blockIdx.y;

    if (c_col >= cols || c_row >= rows) {
        return;
    }

    BLOCK_TYPE new_val = 0;
    BLOCK_TYPE a_val;
    int b_row = BLOCK_SIZE - 1;
    for (int a_i = 0; a_i < cols; ++a_i, b_row += BLOCK_SIZE) {
        a_val = a[c_row * cols + a_i];
        #pragma unroll
        for (int bit = BLOCK_SIZE - 1; bit >= 0; --bit) {
            if ((a_val >> bit & 1)) {
                new_val |= b[(b_row - bit) * cols + c_col];
            }
        }
    }

    BLOCK_TYPE old_val = c[c_row * cols + c_col];
    if (old_val != (old_val | new_val)) {
        _has_changed = true;
        c[c_row * cols + c_col] = new_val | old_val;
    }
}

__global__ void transpose(BLOCK_TYPE *a, BLOCK_TYPE *b, int rows, int cols) {
    int b_col = blockIdx.x * blockDim.x + threadIdx.x;
    int b_row = blockIdx.y;

    if (b_col >= cols || b_row >= rows) {
        return;
    }

    int a_col = b_row / BLOCK_SIZE;
    int a_col_bit = BLOCK_SIZE - 1 - b_row % BLOCK_SIZE;
    int a_row0 = b_col * BLOCK_SIZE;

    BLOCK_TYPE res = 0;
    for (int bit = BLOCK_SIZE - 1; bit >= 0; --bit) {
        if (a_row0 + BLOCK_SIZE - 1 - bit < rows) {
            BLOCK_TYPE old_val = a[(a_row0 + BLOCK_SIZE - 1 - bit) * cols + a_col] >> a_col_bit & 1; 
            res |= old_val << bit;
        }
    }
    b[b_row * cols + b_col] = res;
}

__global__ void _mult_with_add_transpose(BLOCK_TYPE* a, BLOCK_TYPE* b, BLOCK_TYPE* c, int rows, int cols) {
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    int c_row = blockIdx.y;

    if (c_col >= cols || c_row >= rows) {
        return;
    }

    BLOCK_TYPE new_val = 0;
    for (int i = 0; i < cols; ++i) {
        BLOCK_TYPE cur_a = a[c_row * cols + i];
        #pragma unroll
        for (int bit = BLOCK_SIZE - 1; bit >= 0; --bit) {
            if (cur_a & b[(c_col * BLOCK_SIZE + BLOCK_SIZE - 1 - bit) * cols + i]) {
                new_val |= 1U << bit;
            }
        }
    }
    
    BLOCK_TYPE c_old = c[c_row * cols + c_col];
    if (c_old != (new_val | c_old)) {
        _has_changed = true;
        c[c_row * cols + c_col] = new_val | c_old;
    }
}

bool mult_with_add(
    BLOCK_TYPE* a, BLOCK_TYPE* b, BLOCK_TYPE* c, int rows, int cols) {
    BLOCK_TYPE *d_a, *d_b, *d_c;
    int mem_size = rows * cols * sizeof(BLOCK_TYPE);
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_a), mem_size));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_b), mem_size));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_c), mem_size));

    gpuErrchk(cudaMemcpy(d_a, a, mem_size, cudaMemcpyHostToDevice));    
    gpuErrchk(cudaMemcpy(d_b, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_c, c, mem_size, cudaMemcpyHostToDevice));

    bool has_changed = false;
    gpuErrchk(cudaMemcpyToSymbol(_has_changed, &has_changed, sizeof(bool)));

    dim3 blocks(cols / THREADS_NUMBER + 1, rows);
    dim3 threads(THREADS_NUMBER);
    
    _mult_with_add<<<blocks, threads>>>(d_a, d_b, d_c, rows, cols);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // BLOCK_TYPE *d_bT;
    // gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_bT), mem_size));
    // transpose<<<blocks, threads>>>(d_b, d_bT, rows, cols);
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
    // _mult_with_add_transpose<<<blocks, threads>>>(d_a, d_bT, d_c, rows, cols);
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(c, d_c, mem_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpyFromSymbol(&has_changed, _has_changed, sizeof(bool)));

    return has_changed;
}
