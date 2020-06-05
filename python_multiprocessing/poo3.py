import multiprocessing as mp
import numpy as np


b = np.ones((100000,100))

for i in xrange(100): b[1000*i:1000*(i+1),:] = np.ones((1000,100))*i

def task(b, n, i):
    return b[1000*i:1000*(i+1),:]*n

p = mp.Pool(processes=200)

c = np.zeros((100000,100))


for i in xrange(100):
    res = p.apply(task, (b, np.arange(100), i))

#for i in xrange(100): c[1000*i:1000*(i+1),:] = res[i]
