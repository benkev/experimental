#
# Find the minimal conex hull for a set of n points on a plane
# The main loop starts right from the two extremal points
# New plotting and printing routines
#
from pylab import *
import sys
import copy

def plot_dots(ix, r):
    N = r.shape[0]
    print 'ix = ', ix
    for i in xrange(N):
        if ix[i] == -1: plot(r[i,0], r[i,1], 'b.')
        if ix[i] ==  2: plot(r[i,0], r[i,1], 'ro') # Border points
        if ix[i] ==  0: plot(r[i,0], r[i,1], 'g.')
        if ix[i] ==  1: plot(r[i,0], r[i,1], 'm.')

def plot_border(bd, r):
    nbd = len(bd)
    N = r.shape[0]
    for i in xrange(nbd):
        i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
        plot((r[bd[i],0], r[bd[i1],0]), (r[bd[i],1], r[bd[i1],1]), 'b')
        axis('equal'); grid(1)

def print_border(bd, r):
    nbd = len(bd)
    N = r.shape[0]
    for i in xrange(nbd):
        i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
        print bd[i], r[bd[i],:], '->', bd[i1], r[bd[i1],:]
    print 'bd = ', bd


figure()

N = 1000
r = randn(N,2)  # All the points (x,y)
ix = empty(N,dtype=int)
#
# Seed all the points with (-1)s ------------------------------------------
#
ix[:] = -1

#
# Find 2 initial points. They will be on the border.
#
imin = argmin(r[:,1])
imax = argmax(r[:,1])
ix[imin] = 2
ix[imax] = 2
#
# Form initial border vertex list from the two extremal points.
# The border now consists of two (coincident) segments
#
bd = [imin, imax]   # List of border elements
pbd = 0              # List pointer

#
# Main loop
#
extr = True
ncyc = 0

while extr:
    nbd = len(bd)
    print ncyc, ': bd = ', bd, ', nbd = ', nbd
    bd1 = copy.copy(bd)
    pbd = 0 # Pointer to the end of segment
    # Seed all the non-zero and non-2 points with -1
    #for i in xrange(N):
    #    if (ix[i] != 0) and (ix[i] != 2): ix[i] = -1
    for i in xrange(nbd):
        pbd = pbd + 1
        i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
        seg = r[bd[i1],:] - r[bd[i],:]        # Segment vector
        lseg = vector_lengths(seg) # Its length as sqrt(sum(seg**2))
        dmaxr = -1.0  # Right distance can only be positive
        for j in xrange(N):
            if ix[j] == 0: continue # Avoid interior (ix=0) 
            if ix[j] == 2: continue # Avoid border (ix=2) 
            pt = r[j,:] - r[bd[i],:] # Vector from 0-th vertex to a j-th point
            lpt = vector_lengths(pt)
            # Left(-) or right(+) position of j-th point wrt to i-th segment
            psinph = (seg[1]*pt[0] - seg[0]*pt[1])/lseg  # lpt*sin(phi)
            if psinph >= 0.0:
                ix[j] = 1 # Mark r[j,:] as exterior point
                if psinph > dmaxr:  # Ext. point farthest from seg.
                    dmaxr = psinph
                    jmaxr = j
        if dmaxr >= 0: # Farthest exterior point found
            bd1.insert(pbd,jmaxr) # Insert new vort. bw i-th and (i+1)-th
            pbd = pbd + 1
            ix[jmaxr] = 2  # Mark as border point
    extr = False
    for j in xrange(N):
        if ix[j] == -1: ix[j] = 0
        if ix[j] ==  1:
            extr = True
            ix[j] = -1 # The points outside the current pentagon
    bd = bd1
    ncyc = ncyc + 1
    plot_border(bd1, r)



#
# Plot new border
#
print 'New border:'
plot_dots(ix, r)
print_border(bd, r)
show()
