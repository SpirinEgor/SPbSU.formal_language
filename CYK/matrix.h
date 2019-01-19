#ifndef CYK_MATRIX_H
#define CYK_MATRIX_H

typedef unsigned long long ull;

#include <iostream>
#include <vector>
#include <cassert>

class Matrix {
 public:
    size_t n_rows, n_columns, real_size;

    explicit Matrix(size_t n_vertices) : real_size(n_vertices) {
        n_vertices = static_cast<size_t>(n_vertices < 64 ? 64 : find_closest_bin_power(n_vertices));
        n_rows = n_vertices;
        n_columns = static_cast<size_t>(n_vertices / 64);
        matrix.resize(static_cast<size_t>(n_rows));
        for (auto& row : matrix) {
            row.resize(n_columns, 0);
        }
    }

    int find_closest_bin_power(int n) {
        int p = 0, orig = n;
        while ((n >>= 1) != 0) {
            ++p;
        }
        int result = (1 << p);
        return (result < orig ? result * 2 : result);
    }

    bool set(int row, int col, ull value) {
        if (row < 0 || row >= n_rows || col < 0 || col > n_columns) {
            return false;
        }
        matrix[row][col] = value;
        return true;
    }

    bool set(int row, int col, bool value) {
        if (row < 0 || row >= n_rows || col < 0 || (col / 64) > n_columns) {
            return false;
        }
        int real_col = col / 64,
            bit = 63 - col % 64;
        matrix[row][real_col] |= static_cast<ull>(value) << bit;
        return true;
    }

    ull get(int row, int col) const {
        if (row < 0 || row >= n_rows || col < 0 || col > n_columns) {
            throw std::invalid_argument("matrix indexes out of bound");
        }
        return matrix[row][col];
    }

    const std::vector<ull>& get_row(int row) const {
        if (row < 0 || row >= n_rows) {
            throw std::invalid_argument("row out of bound");
        }
        return matrix[row];
    }

    bool set_all(const std::vector<ull>& values) {
        if (values.size() != n_columns * n_rows) {
            throw std::invalid_argument("Not enough value for inserted in matrix");
        }
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_columns; ++c) {
                if (!set(r, c, values[r * n_columns + c])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool set_with_compress(const std::vector<bool>& values) {
        std::vector<ull> compressed;
        for (int ind = 0; ind < values.size(); ind += 64) {
            ull cur_val = 0;
            for (int bit = 63; bit >= 0; --bit) {
                if (ind + 63 - bit == values.size()) {
                    break;
                }
                cur_val |= static_cast<ull>(values[ind + 63 - bit]) << bit;
            }
            compressed.push_back(cur_val);
        }
        compressed.resize(n_rows * n_columns, 0);
        return set_all(compressed);
    }

    Matrix transpose() const {
        Matrix result(real_size);
        std::vector<bool> new_values;
        for (int c = 0; c < n_columns; ++c) {
            for (int ind = 63; ind >= 0; --ind) {
                for (int r = 0; r < n_rows; ++r) {
                    new_values.push_back((get(r, c) >> ind & 1) != 0);
                }
            }
        }
        result.set_with_compress(new_values);
        return result;
    }

    bool operator==(const Matrix& other) const {
        if (n_rows != other.n_rows || n_columns != other.n_columns || real_size != other.real_size) {
            return false;
        }
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_columns; ++c) {
                if (get(r, c) != other.get(r, c)) {
                    return false;
                }
            }
        }
        return true;
    }

    void print() {
        for (int r = 0; r < min(n_rows, real_size); ++r) {
            for (int c = 0; c < n_columns; ++c) {
                ull cur_val = get(r, c);
                for (int ind = 63; ind >= 0; --ind) {
                    if (c * 64 + (63 - ind) >= real_size) {
                        break;
                    }
                    std::cout << (cur_val >> ind & 1);
                }
                std::cout << ' ';
            }
            std::cout << '\n';
        }
        std::cout << std::endl;
    }

    void print_size() const {
        std::cout << n_rows << " x " << n_columns << std::endl;
    }



 private:
    std::vector<std::vector<ull>> matrix;
};

#endif   // CYK_MATRIX_H
