#ifndef CYK_READER_H
#define CYK_READER_H

#include <fstream>
#include <string>
#include <unordered_map>

#include "matrix.h"

class Reader {
 public:
    int nonterm_count = 0, num_vertices = 0;
    std::unordered_map<std::string, int> str2num;
    std::unordered_map<std::string, std::vector<int>> reverse_prod;
    std::vector<std::pair<int, std::pair<int, int>>> nonterm_production;
    std::vector<std::pair<std::string, std::pair<int, int>>> edge;

    Reader(char* chomsky_file, char* graph_file) {
        auto chomsky_stream = std::ifstream(chomsky_file, std::ifstream::in);
        std::string head, body1, body2;
        while (chomsky_stream >> head >> body1) {
            if (!str2num.count(head)) {
                str2num[head] = nonterm_count++;
            }
            if (body1[0] >= 'A' && body1[0] <= 'Z') {
                chomsky_stream >> body2;
                if (!str2num.count(body1)) {
                    str2num[body1] = nonterm_count++;
                }
                if (!str2num.count(body2)) {
                    str2num[body2] = nonterm_count++;
                }
                nonterm_production.push_back({str2num[head], {str2num[body1], str2num[body2]}});
            } else {
                if (!reverse_prod.count(body1)) {
                    reverse_prod[body1] = {};
                }
                reverse_prod[body1].push_back(str2num[head]);
            }
        }
        chomsky_stream.close();
        auto graph_stream = std::ifstream(graph_file, std::ifstream::in);
        int from, to;
        while (graph_stream >> from >> body1 >> to) {
            --from, --to;
            edge.push_back({body1, {from, to}});
            num_vertices = std::max(num_vertices, std::max(from, to) + 1);
        }
        graph_stream.close();
    }
};

#endif  // CYK_READER_H
