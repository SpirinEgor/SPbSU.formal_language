import os
from tqdm import tqdm

from .ITest import ISample, ITest

class CircleSample(ISample):
    def __init__(self, n, gen_files):
        super().__init__()
        self.chomsky_file = './circle_test/chomsky.txt'
        self.graph_file = './circle_test/graph_{}.txt'.format(n)
        if not os.path.exists('./circle_test'):
            os.makedirs('./circle_test')

        self.comment = 'graph with {} vertices'.format(n)
        self.graph_size = n

        if gen_files:
            with open(self.chomsky_file, 'w') as f_out:
                lines = [
                    'S a',
                    'S S S',
                    'S S1 S',
                    'S1 S S' 
                ]
                f_out.write('\n'.join(lines))

            with open(self.graph_file, 'w') as f_out:
                for i in range(n):
                    f_out.write('{} a {}\n'.format(i + 1, (i + 1) % n + 1))

    def check_equal(self, answer_file):
        for nonterm in ['S', 'S1']:
            rows = [i + 1 for i in range(self.graph_size) for j in range(self.graph_size)]
            cols = [j + 1 for i in range(self.graph_size) for j in range(self.graph_size)]
            self.result[nonterm] = sorted(zip(rows, cols))
        res = super().check_equal(answer_file)
        del self.result
        return res


class CircleTest(ITest):
    def __init__(self, maxn=5000, total=30, gen_files=True):
        self.name = 'circle test'
        print('Prepare circle tests')
        self.samples = [CircleSample(int(i), gen_files) for i in tqdm(linspace(3, maxn, total), total=total)]

def linspace(start, stop, size):
    step = (stop - start + 1) / (size - 1)
    for i in range(size):
        yield min(start + i * step, stop)
