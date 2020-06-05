import multiprocessing as mp
import numpy as np

def task(n):
    return np.ones((10,10))*n

b = np.ones((100,10))


p = mp.Pool(processes=20)

for i in xrange(10):
    res = p.map(task, np.arange(10), chunksize=1)

for i in xrange(10): b[10*i:10*(i+1),:] = res[i]


