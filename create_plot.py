import os
import subprocess
import time
from tqdm import tqdm

import matplotlib.pyplot as plt

from tests.CircleTest import CircleTest

test = CircleTest(2500, 30, True)
points = []

TIMEOUT = 120

flag = False
for sample in tqdm(test.samples):
    if not flag:
        try:
            start_time = time.time()
            code = subprocess.call(['./run.sh', sample.chomsky_file, sample.graph_file, 'answer.txt'], timeout=TIMEOUT)
            finish_time = time.time()
            if sample.check_equal('answer.txt'):
                points.append((sample.graph_size, finish_time - start_time))
            else:
                print('test #{} ({}): uncorrect'.format(id + 1, sample.comment))
                exit(0)
        except subprocess.TimeoutExpired:
            points.append((sample.graph_size, TIMEOUT))
            flag = True
    else:
        points.append((sample.graph_size, TIMEOUT))


plt.plot(
    list(zip(*points))[0],
    list(zip(*points))[1]
)
plt.xlabel('Number of vertices in graph')
plt.ylabel('Working time or timeout')
if os.path.exists('plot.png'):
    os.remove('plot.png')
plt.savefig('plot.png')
