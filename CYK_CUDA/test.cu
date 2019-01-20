#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "params.h"
#include "multiplication.h"

const int mat_size = 100;
const int test_number = 100;
const bool show = false;

std::mt19937 gen(time(nullptr));
std::uniform_int_distribution<int> dist(0, 1);

int calc_col_number(int n) {
    return n / BLOCK_SIZE + (n % BLOCK_SIZE ? 1 : 0);
}

std::pair<int, int> calc_col_pos(int n) {
    return {
        n / BLOCK_SIZE,
        BLOCK_SIZE - 1 - (n % BLOCK_SIZE)
    };
}

void print(std::vector<bool>& values) {
    for (int i = 0; i < mat_size; ++i) {
        for (int j = 0; j < mat_size; ++j) {
            std::cout << values[i * mat_size + j];
        }
        std::cout << '\n';
    }
}

void print(BLOCK_TYPE* values, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            for (int bit = BLOCK_SIZE - 1; bit >= 0; --bit) {
                std::cout << (values[r * cols + c] >> bit & 1);
            }
            std::cout << ' ';
        }
        std::cout << '\n';
    }
}

void simple_mult(std::vector<bool>& a, std::vector<bool>&b,
                 std::vector<bool>* c) {
    for (int a_r = 0; a_r < mat_size; ++a_r) {
        for (int a_c = 0; a_c < mat_size; ++a_c) {
            for (int b_c = 0; b_c < mat_size; ++b_c) {
                c->at(a_r * mat_size + b_c) = c->at(a_r * mat_size + b_c) || a[a_r * mat_size + a_c] && b[a_c * mat_size + b_c];
            }
        }
    }
}

int main() {
    bool result = true;
    for (int test = 0; test < test_number; ++test) {
        std::vector<bool> val1, val2, res;
        for (int i = 0; i < mat_size * mat_size; ++i) {
            val1.push_back(dist(gen));
            val2.push_back(dist(gen));
        }
        res.resize(mat_size * mat_size, false);
        simple_mult(val1, val2, &res);

        int rows = mat_size, cols = calc_col_number(mat_size);
        BLOCK_TYPE* a = new BLOCK_TYPE [rows * cols];
        BLOCK_TYPE* b = new BLOCK_TYPE [rows * cols];
        BLOCK_TYPE* c = new BLOCK_TYPE [rows * cols];
        std::fill_n(a, rows * cols, 0);
        std::fill_n(b, rows * cols, 0);
        std::fill_n(c, rows * cols, 0);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < mat_size; ++c) {
                auto col_pos = calc_col_pos(c);
                a[r * cols + col_pos.first] |= static_cast<BLOCK_TYPE>(val1[r * mat_size + c]) << col_pos.second;
                b[r * cols + col_pos.first] |= static_cast<BLOCK_TYPE>(val2[r * mat_size + c]) << col_pos.second;
            }
        }

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

        dim3 blocks(cols / 32 + 1, rows);
        dim3 threads(32);
        mult<<<blocks, threads>>>(d_a, d_b, d_c, rows, cols);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaMemcpy(c, d_c, mem_size, cudaMemcpyDeviceToHost));

        for (int r = 0; r < rows; ++r) {
            for (int _c = 0; _c < mat_size; ++_c) {
                auto col_pos = calc_col_pos(_c);
                bool check_value = c[r * cols + col_pos.first] >> col_pos.second & 1;
                result &= (check_value == res[r * mat_size + _c]);
            }
        }

        if (show) {
            std::cout << "A true" << std::endl;
            print(val1);
            std::cout << "B true" << std::endl;
            print(val2);
            std::cout << "True" << std::endl;
            print(res);
            std::cout << "A compress" << std::endl;
            print(a, rows, cols);
            std::cout << "B compress" << std::endl;
            print(b, rows, cols);
            std::cout << "C GPU" << std::endl;
            print(c, rows, cols);
        }
        free(a); free(b); free(c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }
    std::cout << (result ? "Success!!" : "Failure!!") << std::endl;
    return 0;
}
