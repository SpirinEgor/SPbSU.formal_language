#ifndef CYK_MULT_H
#define CYK_MULT_H

#include <cassert>
#include <vector>

#include "matrix.h"

// c = a * b
void multiplication(const Matrix& a, const Matrix& b, Matrix* c) {
    std::vector<bool> new_values;
    for (int a_r = 0; a_r < a.n_rows; ++a_r) {
        for (int b_r = 0; b_r < b.n_rows; ++b_r) {
            auto a_row = a.get_row(a_r);
            auto b_row = b.get_row(b_r);
            bool new_val = false;
            for (int col = 0; col < a.n_columns; ++col) {
                new_val |= (a_row[col] & b_row[col]);
            }
            new_values.push_back(new_val);
        }
    }
    c->set_with_compress(new_values);
}

// c += a * b
// true -- if smth changes
bool mult_with_add(const Matrix& a, const Matrix& b, Matrix* c) {
    Matrix c_mult(c->real_size);
    multiplication(a, b.transpose(), &c_mult);
    bool has_changed = false;
    for (int row = 0; row < c->n_rows; ++row) {
        for (int col = 0; col < c->n_columns; ++col) {
            ull old_value = c->get(row, col),
                new_value = c_mult.get(row, col);
            has_changed |= (old_value != (new_value | old_value));
            if (has_changed) {
                c->set(row, col, new_value | old_value);
            }
        }
    }
    return has_changed;
}


bool check_mult(std::vector<bool>& val1, std::vector<bool>& val2, Matrix& res, int size) {
    std::vector<bool> true_values, check_values;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            bool cur_val = false;
            for (int k = 0; k < size; ++k) {
                cur_val |= val1[i * size + k] && val2[k * size + j];
            }
            true_values.push_back(cur_val);
        }
    }
    for (int r = 0; r < res.real_size; ++r) {
        for (int c = 0; c < res.n_columns; ++c) {
            ull cur_val = res.get(r, c);
            for (int ind = 63; ind >= 0; --ind) {
                if (c * 64 + 63 - ind >= res.real_size) {
                    break;
                }
                check_values.push_back(static_cast<bool>(cur_val >> ind & 1));
            }
        }
    }
    assert(true_values.size() == check_values.size());
    for (int i = 0; i < true_values.size(); ++i) {
        if (true_values[i] != check_values[i]) {
            return false;
        }
    }
    return true;
}

bool test_multiplication1(int n, bool single = false) {
    srand(static_cast<unsigned int>(time(nullptr)));
    std::vector<bool> test_results;
    bool result = true;
    for (int test_num = (single ? n: 1); test_num <= n; ++test_num) {
        std::vector<bool> m_val1, m_val2, m_true_val1, m_true_val2;
        auto m1 = Matrix(static_cast<size_t>(test_num));
        for (int i = 0; i < m1.n_rows; ++i) {
            for (int j = 0; j < m1.n_rows; ++j) {
                if (j < test_num && i < test_num) {
                    m_true_val1.push_back(rand() % 2);
                    m_true_val2.push_back(rand() % 2);
                }
                m_val1.push_back(static_cast<bool>(
                        j >= test_num || i >= test_num ? false : m_true_val1.back()));
                m_val2.push_back(static_cast<bool>(
                        j >= test_num || i >= test_num ? false : m_true_val2.back()));
            }
        }
        m1.set_with_compress(m_val1);
        auto m2 = Matrix(static_cast<size_t>(test_num));
        m2.set_with_compress(m_val2);
        auto c = Matrix(static_cast<size_t>(test_num));
        multiplication(m1, m2.transpose(), &c);
        result &= check_mult(m_true_val1, m_true_val2, c, test_num);
        if (!result) {
            m1.print();
            m2.print();
            c.print();
            return false;
        }
        test_results.push_back(result);
    }
    return result;
}

bool test_multiplication2(int n, int k) {
    bool result = true;
    for (int i = 0; i < k; ++i) {
        result &= test_multiplication1(n, true);
        if (!result)
            return false;
    }
    return result;
}

#endif  // CYK_MULT_H
