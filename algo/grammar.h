//
// Created by vkutuev on 01.11.18.
//

#ifndef ALGOWITHM4RI_GRAMMAR_H
#define ALGOWITHM4RI_GRAMMAR_H

#include <map>
#include <vector>
#include <string>
#include <m4ri/m4ri.h>
#include "graph.h"


typedef std::vector<std::string> heads;
typedef std::pair<std::string, std::string> nonterminal_pair;

class Grammar {
    public:
        Grammar();
        ~Grammar();
        void read(const std::string &filename);
        void print();
    std::vector<std::pair<std::string, std::vector<int>>> intersection_with_graph(int n, graph_t graph);

    private:
        std::map<std::string, mzd_t*> nonterminals;
        std::map<nonterminal_pair, heads> rules;
        std::map<char, std::vector<std::string>> terminals;
};



#endif //ALGOWITHM4RI_GRAMMAR_H
