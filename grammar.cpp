//
// Created by vkutuev on 01.11.18.
//

#include <fstream>
#include <iostream>
#include "grammar.h"
using namespace std;

Grammar::Grammar() = default;

void Grammar::read(const string &filename) {
    ifstream input(filename);

    string line ;
    while (getline(input, line)) {
        unsigned long divider = line.find(' ');
        string head = line.substr(0, divider);
        string body = line.substr(divider + 1, line.size() - (divider + 1));
        nonterminals[head] = nullptr;
        divider = body.find(' ');
        if (divider == string::npos) {
            terminals[body].push_back(head);
            continue;
        }
        string fst = body.substr(0, divider);
        string snd = body.substr(divider + 1, body.size() - (divider + 1));
        nonterminals[fst] = nullptr;
        nonterminals[snd] = nullptr;
        rules[nonterminal_pair(fst, snd)].push_back(head);
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

void Grammar::intersection_with_graph(int n, graph_t graph, char* filename) {
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

    ofstream outputfile;
    outputfile.open(filename);
    for (auto &nonterminal: nonterminals) {
        outputfile << nonterminal.first;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (mzd_read_bit(nonterminal.second, i, j) != 0) {
                    outputfile << ' ' << i << ' ' << j;
                }
            }
        }
        outputfile << endl;
    }
    outputfile.close();
}
Grammar::~Grammar() {
    for (auto &nonterminal: nonterminals) {
        mzd_free(nonterminal.second);
    }
}
