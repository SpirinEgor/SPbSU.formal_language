import os

from .ITest import ISample, ITest

class SimpleSample(ISample):
    def __init__(self):
        super().__init__()
        self.chomsky_file = './simple_test/chomsky.txt'
        self.graph_file = './simple_test/graph.txt'
        if not os.path.exists('./simple_test'):
            os.makedirs('./simple_test')

        self.comment = 'test from example'

        self.graph_size = 4
        correct = [
            'L', 
            'R', 
            'S2 2 1 2 3',
            'S1 2 2 2 4',
            'E 1 2 1 4 1 2',
            'N'
        ]
        for line in correct:
            line_split = line.split(' ')
            s = line_split[0]
            if len(line_split) == 1:
                res = []
            else:
                rows = map(int, line_split[1::2])
                cols = map(int, line_split[2::2])
                res = sorted(zip(rows, cols))
            self.result[s] = res
        
        with open(self.chomsky_file, 'w+') as f_out:
            lines = [
                'S2 *',
                'S2 +',
                'L (',
                'R )'
                'E num',
                'E E S1',
                'E L N',
                'S1 S2 E',
                'N E R',
            ]
            f_out.write('\n'.join(lines))
        with open(self.graph_file, 'w+') as f_out:
            lines = [
                '1 num 2',
                '2 + 3',
                '3 num 4',
                '2 * 1'
            ]
            f_out.write('\n'.join(lines))
                           

class SimpleTest(ITest):
    def __init__(self):
        self.name = 'simple test'
        self.samples = [SimpleSample()]