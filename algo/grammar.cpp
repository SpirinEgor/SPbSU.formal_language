//
// Created by vkutuev on 01.11.18.
//

#include <fstream>
#include <iostream>
#include "grammar.h"
using namespace std;

int matrix_size;
void set_matrix_size(int size) {
    matrix_size = size;
}

Grammar::Grammar() = default;

void Grammar::read(const string &filename) {
    ifstream input(filename);

    string line ;
    while (getline(input, line)) {
        unsigned long divider = (line.find(':'));
        string head = line.substr(0, divider);
        string body = line.substr(divider + 1, line.size() - (divider + 1));
        nonterminals[head] = nullptr;
        if (body[0] != toupper(body[0]) || !isalpha(body[0])) {
            terminals[body[0]].push_back(head);
            continue;
        }
        for (int i = 1; i < body.size(); ++i) {
            if (isalpha(body[i]) && body[i] == toupper(body[i])) {
                string fst = body.substr(0, i);
                string snd = body.substr(i, body.size() - 1);
                nonterminals[fst] = nullptr;
                nonterminals[snd] = nullptr;
                rules[nonterminal_pair(fst, snd)].push_back(head);
                break;
            }
        }
    }
    input.close();
}

void Grammar::print() {
    cout << "Nonterminals: ";
    for (auto &nonterminal : nonterminals) {
        cout << nonterminal.first << ' ';
    }
    cout << '\n';
    cout << "Terminals: ";
    for (auto &terminal : terminals) {
        cout << terminal.first << ' ';
    }
    cout << '\n';
    cout << "Prods:\n";
    for (auto &prod : rules) {
        cout << prod.first.first << ' ' << prod.first.second << " <- ";
        for (auto &head : prod.second) {
            cout << head << ", ";
        }
       cout << '\n';
    }
    cout << '\n';
}

vector<pair<string, vector<int>>> Grammar::intersection_with_graph(int n, graph_t graph) {
    for (auto &nonterminal: nonterminals) {
        nonterminal.second = mzd_init(n, n);
    }

    for (auto &edge: graph) {
        for (auto &nonterm: terminals[edge.label]) {
            mzd_write_bit(nonterminals[nonterm], edge.from, edge.to, 1);
        }
    }

    bool finished = false;

    while (!finished) {
        finished = true;
        for (auto &rule: rules) {
            mzd_t* mul_result = mzd_sr_mul_m4rm(nullptr, nonterminals[rule.first.first], nonterminals[rule.first.second], 0);
            for (auto &head: rule.second) {
                mzd_t* new_head = mzd_or(nullptr, nonterminals[head], mul_result);
                if (!mzd_equal(nonterminals[head], new_head)) {
                    mzd_free(nonterminals[head]);
                    nonterminals[head] = new_head;
                    finished = false;
                } else {
                    mzd_free(new_head);
                }
            }
            mzd_free(mul_result);
        }
    }

    vector<pair<string, vector<int>>> answer;// = vector<vector<string>>(nonterminals.size());
    for (auto &nonterminal: nonterminals) {
        vector<int> positions;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (mzd_read_bit(nonterminal.second, i, j) != 0) {
                    positions.push_back(i);
                    positions.push_back(j);
                }
            }
        }
        answer.emplace_back(nonterminal.first, positions);
    }

    return answer;
}
Grammar::~Grammar() {
    for (auto &nonterminal: nonterminals) {
        mzd_free(nonterminal.second);
    }
}
