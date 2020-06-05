import multiprocessing as mp
import numpy as np


def task(n):
    return np.ones((1000,100))*n

b = np.ones((100000,100))

p = mp.Pool(processes=200)


res = p.map(task, np.arange(100), chunksize=1)

for i in xrange(100): b[1000*i:1000*(i+1),:] = res[i]
