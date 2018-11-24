import os

from .ITest import ISample, ITest

class SimpleSample(ISample):
    def __init__(self):
        self.chomsky_file = './simple_test/chomsky.txt'
        self.graph_file = './simple_test/graph.txt'
        self.result_file = './simple_test/result.txt'
        if not os.path.exists('./simple_test'):
            os.makedirs('./simple_test')

        self.comment = 'test from example'

        self.graph_size = 4
        self.correct_matrix = {}
        matrix = [
            '1 2 E',
            '1 4 E',
            '2 1 S2',
            '2 2 S1',
            '2 3 S2',
            '2 4 S1',
            '3 4 E'
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
                '1 2 num',
                '2 3 +',
                '3 4 num',
                '2 1 *'
            ]
            f_out.write('\n'.join(lines))
        with open(self.result_file, 'w+') as f_out:
            f_out.write('\n'.join(matrix))

    def check_equal(self, answer_file):
        if not os.path.exists(answer_file):
            return False
        answer_matrix = {}
        with open(answer_file, 'r') as f_in:
            for line in f_in:
                line_split = line.split(' ')
                row = int(line_split[0])
                col = int(line_split[1])
                ss = line_split[2:]
                if ss[-1] == '\n':
                    ss = ss[:-1]
                if row not in answer_matrix:
                    answer_matrix[row] = {}
                answer_matrix[row][col] = ss
        return answer_matrix == self.correct_matrix
                           

class SimpleTest(ITest):
    def __init__(self):
        self.name = 'simple test'
        self.samples = [SimpleSample()]