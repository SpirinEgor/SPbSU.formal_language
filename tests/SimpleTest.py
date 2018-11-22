import os

from .ITest import ISample, ITest

class SimpleSample(ISample):
    def __init__(self):
        self.chomsky_file = './simple_test/chomsky.txt'
        self.graph_file = './simple_test/graph.txt'
        self.result_file = './simple_test/result.txt'
        if not os.path.exists('./simple_test'):
            os.makedirs('./simple_test')

        self.graph_size = 4
        self.correct_matrix = {}
        matrix = [
            '0 1 E',
            '3 0 E',
            '1 0 S2',
            '1 1 S1',
            '1 2 S2',
            '1 3 S1',
            '2 3 E'
        ]
        for line in matrix:
            row, col, s = line.split(' ')
            if int(row) not in self.correct_matrix:
                self.correct_matrix[int(row)] = {}
            self.correct_matrix[int(row)][int(col)] = [s]

        with open(self.chomsky_file, 'w+') as f_out:
            lines = [
                'E : E S1',
                'E : num',
                'E : L N',
                'S1 : S2 E',
                'S2 : +',
                'S2 : *',
                'L : (',
                'N : E R',
                'R : )'
            ]
            f_out.write('\n'.join(lines))
        with open(self.graph_file, 'w+') as f_out:
            lines = [
                '0 1 num',
                '1 2 +',
                '2 3 num',
                '1 0 *'
            ]
            f_out.write('\n'.join(lines))
        with open(self.result_file, 'w+') as f_out:
            f_out.write('\n'.join(matrix))

    def check_equal(self, answer_file):
        if not os.path.exists(answer_file):
            return False
        self.answer_matrix = {}
        with open(answer_file, 'r') as f_in:
            for line in f_in:
                line_split = line.split(' ')
                row = int(line_split[0])
                col = int(line_split[1])
                ss = line_split[2:]
                if row not in line_split:
                    line_split[row] = {}
                line_split[row][col] = ss
        return self.answer_matrix == self.correct_matrix
                           

class SimpleTest(ITest):
    def __init__(self):
        self.name = 'simple test'
        self.samples = [SimpleSample()]