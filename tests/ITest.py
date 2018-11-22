class ISample:
    def __init__(self):
        self.chomsky_file: str = 'Chomsky'
        self.graph_file: str = 'Graph'
        self.result_file: str = 'Result'

        self.comment = 'interface'

    def check_equal(self, answer_file) -> bool:
        return True

class ITest:
    def __init__(self):
        self.name: str = 'ITest'
        self.samples = [ISample()]
