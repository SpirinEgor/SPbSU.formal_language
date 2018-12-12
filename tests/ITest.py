class ISample:
    def __init__(self):
        self.chomsky_file: str = 'Chomsky'
        self.graph_file: str = 'Graph'

        self.comment = 'interface'
        self.result = {}

    def check_equal(self, answer_file) -> bool:
        answer = {}
        with open(answer_file, 'r') as f_in:
            for line in f_in:
                line_split = line.split(' ')
                if line_split[-1] in ['', '\n']:
                    line_split = line_split[:-1]
                if len(line_split) == 1:
                    answer[line_split[0]] = []
                    continue
                nonterm = line_split[0]
                rows = map(int, line_split[1::2])
                cols = map(int, line_split[2::2])
                answer[nonterm] = sorted(zip(rows, cols))
        # print(answer)
        # print(self.result)
        for nonterm in answer:
            if nonterm not in self.result or answer[nonterm] != self.result[nonterm]:
                return False
        return len(self.result) == len(answer)

class ITest:
    def __init__(self):
        self.name: str = 'ITest'
        self.samples = [ISample()]
