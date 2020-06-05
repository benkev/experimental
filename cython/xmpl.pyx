from numpy import *

def example(a, b):
    na = len(a);
    mb = size(b,0)
    nb = size(b,1)
    for i in xrange(na):
        print a[i]

    print

    for i in xrange(mb):
        for j in xrange(nb):
            print b[i,j]
     
    a[:] = 0.
