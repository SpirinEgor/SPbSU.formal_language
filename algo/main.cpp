#include "graph.h"
#include <fstream>
#include <iostream>
#include "grammar.h"
#include <ctime>

using namespace std;

int main() {

    auto * grammar = new Grammar;
    grammar->read("grammar.txt");
    grammar->print();

    graph_t graph = graph_t();
    int n = read_graph(const_cast<char *>("graph.txt"), graph);
//    print_graph(graph);

    clock_t begin = clock();
    vector<pair<string, vector<int>>> ans = grammar->intersection_with_graph(n, graph);
    clock_t end = clock();
    for (auto &line : ans) {
        cout << line.first << ' ';
        for (const auto &item: line.second) {
            cout << item << ' ';
        }
        cout << endl;
    }

    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    delete grammar;


    cout << "Passed: " << elapsed_secs << "sec" << endl;

    return 0;
}