//
// Created by vkutuev on 01.11.18.
//

#include "graph.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int read_graph(char* filename, graph_t& graph) {
    FILE* file = fopen(filename, "r");
    int n = 0;
    edge_t edge;
    while (fscanf(file, "%d %s %d", &edge.from, edge.label, &edge.to) != EOF) {
//        cout << edge.from << ' ' << edge.to << ' ' << edge.label << endl;
        n = max(n, edge.from);
        n = max(n, edge.to);
        graph.push_back(edge);
    }
    fclose(file);
    return n + 1;
}

void print_graph(graph_t graph) {
    for (auto &edge: graph)
        cout << edge.from << ' ' << edge.to << " [" << edge.label << ']' << endl;
}

