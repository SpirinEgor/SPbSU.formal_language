#include "graph.h"
#include <fstream>
#include <iostream>
#include "grammar.h"
#include <ctime>

using namespace std;

int main(int argc, char* argv[]) {

    auto * grammar = new Grammar;
    grammar->read(argv[1]);
//    grammar->print();

    graph_t graph = graph_t();
    int n = read_graph(argv[2], graph);
//    print_graph(graph);

//    clock_t begin = clock();
    grammar->intersection_with_graph(n, graph, argv[3]);
//    clock_t end = clock();

//    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    delete grammar;


//    cout << "Passed: " << elapsed_secs << "sec" << endl;

    return 0;
}