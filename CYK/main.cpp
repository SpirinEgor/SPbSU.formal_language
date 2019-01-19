#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

#include "matrix.h"
#include "reader.h"
#include "mult.h"



int main(int argc, char* argv[]) {
    auto reader = Reader(argv[1], argv[2]);

    vector<Matrix> mat;
    mat.reserve(static_cast<unsigned long>(reader.nonterm_count));
    for (int i = 0; i < reader.nonterm_count; ++i) {
        mat.emplace_back(reader.num_vertices);
    }

    for (auto& edge : reader.edge) {
        for (int nonterm : reader.reverse_prod[edge.first]) {
            mat[nonterm].set(edge.second.first, edge.second.second, true);
        }
    }

    while (true) {
        bool has_changed = false;
        for (auto& prod : reader.nonterm_production) {
            has_changed |= mult_with_add(mat[prod.second.first], mat[prod.second.second], &mat[prod.first]);
        }
        if (!has_changed) {
            break;
        }
    }

    auto out_stream = ofstream(argv[3], std::ofstream::out);
    for (auto& nonterm : reader.str2num) {
        out_stream << nonterm.first << ' ';
        for (int r = 0; r < mat[nonterm.second].real_size; ++r) {
            for (int c = 0; c < mat[nonterm.second].n_columns; ++c) {
                ull cur_val = mat[nonterm.second].get(r, c);
                for (int bit = 63; bit >= 0; --bit) {
                    if (c * 64 + 63 - bit >= mat[nonterm.second].real_size) {
                        break;
                    }
                    if (cur_val >> bit & 1) {
                        out_stream << r + 1 << ' ' << c * 64 + 63 - bit + 1 << ' ';
                    }
                }
            }
        }
        out_stream << endl;
    }
    out_stream.close();

    return 0;
}
