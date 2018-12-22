//
// Created by vkutuev on 01.11.18.
//

#include "graph.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int read_graph(char* filename, graph_t& graph) {
    FILE* file = fopen(filename, "r");
    int n;
    edge_t edge;
    for (n = 0; fscanf(file, "%d%d %c", &edge.from, &edge.to, &edge.label) != EOF; ++n) {
//        cout << edge.from << ' ' << edge.to << ' ' << edge.label << endl;
        graph.push_back(edge);
    }
    fclose(file);
    return n;
}

void print_graph(graph_t graph) {
    for (auto &edge: graph)
        cout << edge.from << ' ' << edge.to << ' ' << edge.label << endl;
}

