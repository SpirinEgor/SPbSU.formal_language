#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "params.h"
#include "reader.h"
#include "multiplication.h"


int calc_col_number(int n) {
    return n / BLOCK_SIZE + (n % BLOCK_SIZE ? 1 : 0);
}

std::pair<int, int> calc_col_pos(int n) {
    return {
        n / BLOCK_SIZE,
        BLOCK_SIZE - 1 - (n % BLOCK_SIZE)
    };
}

int main(int argc, char* argv[]) {
    auto reader = Reader(argv[1], argv[2]);

    auto rows = reader.num_vertices;
    auto cols = calc_col_number(reader.num_vertices);
    std::vector<BLOCK_TYPE*> mat(reader.nonterm_count);
    for (int i = 0; i < reader.nonterm_count; ++i) {
        mat[i] = new BLOCK_TYPE [rows * cols];
        std::fill_n(mat[i], rows * cols, 0U);
    }

    for (auto& edge : reader.edge) {
        for (int nonterm : reader.reverse_prod[edge.first]) {
            auto col_pos = calc_col_pos(edge.second.second);
            mat[nonterm][edge.second.first * cols + col_pos.first] |= 1U << col_pos.second;
        }
    }

    while (true) {
        bool has_changed = false;
        for (auto& prod : reader.nonterm_production) {
            has_changed |= mult_with_add(mat[prod.second.first], mat[prod.second.second], mat[prod.first], rows, cols);
        }
        if (!has_changed) {
            break;
        }
    }

    auto output_stream = std::ofstream(argv[3], std::ofstream::out);
    for (auto& nonterm : reader.str2num) {
        output_stream << nonterm.first << ' ';
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                for (int bit = BLOCK_SIZE - 1; bit >= 0; --bit) {
                    if (mat[nonterm.second][r * cols + c] >> bit & 1) {
                        output_stream << r << ' ' << c * BLOCK_SIZE + BLOCK_SIZE - bit - 1 << ' ';
                    }
                }
            }
        }
        output_stream << std::endl;
    }
    return 0;
}