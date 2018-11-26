from collections import defaultdict
from copy import deepcopy
import sys
#
vertices = 0

def read():
    with open(sys.argv[1]) as grammar_file, open(sys.argv[2]) as graph_file:
        grammar_term = defaultdict(set)
        grammar_vars = defaultdict(set)
        for line in grammar_file:
            rule_symbol_index = line.find(':')
            head = line[:rule_symbol_index].strip()
            tail = line[rule_symbol_index + 2:].strip().split(' ')
            if len(tail) == 1:
                grammar_term[tail[0]].add(head)
            else:
                grammar_vars[(tail[0], tail[1])].add(head)

        source = 'source'
        dest = 'dest'
        symb = 'symb'

        edges = list(map(
            lambda l: {source: int(l[0]), dest: int(l[1]), symb: l[2]},
            [line.split() for line in graph_file]
        ))

        global vertices
        vertices = max(edge[index] for edge in edges for index in [source, dest]) + 1

        graph = [[0] * vertices for i in range(vertices)]

        for edge in edges:
            graph[edge[source]][edge[dest]] = edge[symb]

    return grammar_term, grammar_vars, graph


def initialize(grammar_term, graph):
    for line in graph:
        for element in range(vertices):
            line[element] = deepcopy(grammar_term.get(line[element])) or set()


def make_new(grammar_vars, old, level):
    new = deepcopy(old)
    for i in range(vertices):
        for j in range(vertices):
            for k in range(vertices):
                for a_ik in old[i][k]:
                    for b_kj in old[k][j]:
                        for t in grammar_vars[(a_ik, b_kj)]:
                            new[i][j].add(t)
    return new


def print_graph(graph):
    mx = 4
    tmps = ((mx + 1) * (vertices) + 1) * '_'
    print(tmps)
    for line in graph:
        print('|', end='')
        for t in line:
            t = ','.join(t)
            x = str(t)
            ma = max(mx - len(x), 0)
            tmp = ' ' * ma
            print(tmp + x, end='|')
        print()
        print(tmps)
    print()

def print_graph_egor(graph):
    with open(sys.argv[3], 'w') as writer:
        output = []
        for i, line in enumerate(graph):
            for j, element in enumerate(line):
                if element:
                    output.append('{0} {1} {2}'.format(i, j, ' '.join(element)))
        writer.write('\n'.join(output))

def main():
    grammar_term, grammar_vars, old_graph = read()
    initialize(grammar_term, old_graph)
    new_graph = old_graph
    old_graph = None
    level = 0

    while old_graph != new_graph:
        old_graph = deepcopy(new_graph)
        new_graph = make_new(grammar_vars, new_graph, level)
        level += 1

    print_graph_egor(new_graph)


if __name__ == '__main__':
    main()
