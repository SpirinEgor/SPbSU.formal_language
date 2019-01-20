#include <iostream>
#include <fstream>

#include "reader.h"

int main(int argc, char* argv[]) {
    auto reader = Reader(argv[1], argv[2]);

    auto output_stream = std::ofstream(argv[3], std::ofstream::out);
    output_stream << reader.nonterm_count << " " << reader.num_vertices << std::endl;
    return 0;
}