import subprocess
import time
from typing import Generator

from tests.ITest import ITest
from tests.SimpleTest import SimpleTest

tests = [ITest(), SimpleTest()]

for test in tests:
    print('=== {} ==='.format(test.name))
    for id, sample in enumerate(test.samples):
        start_time = time.time()
        subprocess.call(['./run.sh', sample.chomsky_file, sample.graph_file, 'answer.txt'])
        finish_time = time.time()
        if sample.check_equal('answer.txt'):
            print('test #{} done in {}ms'.format(id + 1, finish_time - start_time))
        else:
            print('test #{}: uncorrect'.format(id + 1))
            break
