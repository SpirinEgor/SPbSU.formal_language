import subprocess
import time
from typing import Generator

from tests.ITest import ITest

tests = [ITest()]

for test in tests:
    total_time = 0
    correct = True
    for sample in test.samples:
        start_time = time.time()
        subprocess.call(['./run.sh', sample.chomsky_file, sample.graph_file, 'answer.txt'])
        finish_time = time.time()
        total_time += (finish_time - start_time)
        if not sample.check_equal('answer.txt'):
            correct = False
            break
    if correct:
        print('test: {} done in {}ms'.format(test.name, total_time / len(test.samples)))
    else:
        print('test: {} uncorrect'.format(test.name))
