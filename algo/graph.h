//
// Created by vkutuev on 01.11.18.
//

#ifndef ALGOWITHM4RI_GRAPH_H
#define ALGOWITHM4RI_GRAPH_H

//#include <m4ri/mzd.h>
#include <string>
#include <vector>

typedef struct edge_s {
    int from;
    int to;
    char label;
} edge_t;

typedef std::vector<edge_t> graph_t;

int read_graph(char* filename, graph_t& graph);
void print_graph(graph_t graph);

#endif //ALGOWITHM4RI_GRAPH_H
