import os

from .ITest import ISample, ITest

class CircleSample(ISample):
    def __init__(self, n):
        self.chomsky_file = './circle_test/chomsky.txt'
        self.graph_file = './circle_test/graph_{}.txt'.format(n)
        self.result_file = './circle_test/result_{}.txt'.format(n)
        if not os.path.exists('./circle_test'):
            os.makedirs('./circle_test')

        self.comment = 'graph with {} vertices'.format(n)
        self.graph_size = n

        with open(self.chomsky_file, 'w+') as f_out:
            lines = [
                'S : S S',
                'S : S1 S',
                'S : a',
                'S1 : SS' 
            ]
            f_out.write('\n'.join(lines))

        with open(self.graph_file, 'w+') as f_out:
            for i in range(n):
                f_out.write('{} {} a\n'.format(i + 1, (i + 1) % n + 1))
        
        with open(self.result_file, 'w+') as f_out:
            for i in range(n):
                for j in range(n):
                    f_out.write('{} {} S S1\n'.format(i, j))
    
    def check_equal(self, answer_file):
        if not os.path.exists(answer_file):
            return False
        check_matrix = [[0 for _ in range(self.graph_size)]
                            for _ in range(self.graph_size)]
        with open(answer_file, 'r') as f_in:
            for line in f_in:
                line_split = line.split(' ')
                row, col = int(line_split[0]), int(line_split[1])
                ss = line_split[2:]
                if (len(ss) != 2 or not ('S' in ss and 'S1' in ss)):
                    return False
                check_matrix[row - 1][col - 1] = 1
        return sum(check_matrix) == self.graph_size * self.graph_size


class CircleTest(ITest):
    def __init__(self):
        self.name = 'circle test'
        self.samples = [CircleSample(i) for i in [3, 5, 10]]
