#
# Study properties of digital correlation
#

from pylab import *

nx = 3
ny = 4
ix = arange(nx)
iy = arange(ny)

print 'x corr y = Sum x[n]*y[n+m]:\n'

for m in xrange(-10,10):
    im = 0
    for n in xrange(10):
        if n >= 0 and n < nx and n+m >= 0 and n+m < ny:
            p = (ix[n], iy[n+m])
            if im == 0: 
                print 'm = %3d   ' % m,
                im = 1
            print '(%d,%d) ' % p,
    if im == 1: print

print

print 'x corr y = Sum x[n-m]*y[n]:\n'

for m in xrange(-10,10):
    im = 0
    for n in xrange(10):
        if n-m >= 0 and n-m < nx and n >= 0 and n < ny:
            p = (ix[n-m], iy[n])
            if im == 0: 
                print 'm = %3d   ' % m,
                im = 1
            print '(%d,%d) ' % p,
    if im == 1: print

print

print 'y corr x = Sum x[n+m]*y[n]:\n'

for m in xrange(-10,10):
    im = 0
    for n in xrange(10):
        if n+m >= 0 and n+m < nx and n >= 0 and n < ny:
            p = (ix[n+m], iy[n])
            if im == 0: 
                print 'm = %3d   ' % m,
                im = 1
            print '(%d,%d) ' % p,
    if im == 1: print

print

print 'y corr x = Sum x[n]*y[n-m]:\n'

for m in xrange(-10,10):
    im = 0
    for n in xrange(10):
        if n >= 0 and n < nx and n-m >= 0 and n-m < ny:
            p = (ix[n], iy[n-m])
            if im == 0: 
                print 'm = %3d   ' % m,
                im = 1
            print '(%d,%d) ' % p,
    if im == 1: print

print



