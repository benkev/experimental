#
# Find the minimal conex hull for a set of n points on a plane
#
from pylab import *
import sys
import copy

def plot_hull(bd, ix, r):
    nbd = len(bd)
    N = r.shape[0]
    #figure()
    for i in xrange(nbd):
        i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
        plot((r[bd[i],0], r[bd[i1],0]), (r[bd[i],1], r[bd[i1],1]), 'c')
        print bd[i], r[bd[i],:], '--', bd[i1], r[bd[i1],:]
    print 'bd = ', bd
    print 'ix = ', ix

    for i in xrange(N):
        if ix[i] == -1: plot(r[i,0], r[i,1], 'b.')
        if ix[i] ==  2: plot(r[i,0], r[i,1], 'ro') # Border points
        if ix[i] ==  0: plot(r[i,0], r[i,1], 'g.')
        if ix[i] ==  1: plot(r[i,0], r[i,1], 'm.')


N = 100
r = randn(N,2)  # All the points (x,y)
ix = empty(N,dtype=int)
#
# Seed all the points with (-1)s ------------------------------------------
#
ix[:] = -1
#print 'r =', r 
figure()
plot(r[:,0], r[:,1], 'k.'); grid(1); axis('equal')

#
# Find 2 initial points. They will be on the border.
#
imin = argmin(r[:,1])
imax = argmax(r[:,1])
ix[imin] = 2
ix[imax] = 2
print 'imin = ', imin
print 'imax = ', imax
plot((r[imin,0], r[imax,0]),(r[imin,1], r[imax,1]), 'k')
#
# Form initial border vertex list from the two extremal points.
# The border now consists of two (coincident) segments
#
bd = [imin, imax]   # List of border elements
pbd = 0              # List pointer
abd = array(r[bd])

#
# Find the points farthest from the segments ON THE RIGHT of both
# The product of vectors (bd[0],bd[1]) and (bd[0],r[j]) is used.
#
# From 0 to 1:
#
seg = r[bd[1],:] - r[bd[0],:]        # Segment vector
lseg = vector_lengths(seg) # Its length as sqrt(sum(seg**2))
dmaxr = -1.0  # Right distance can only be positive
dmaxl = 1.0  # Right distance can only be negative
jmaxr = -1
jmaxl = -1
for j in xrange(N):
    if (j == imin) or (j == imax): continue     #skip the border points
    print 'j = ', j
    pt = r[j,:] - r[bd[0],:] # Vector from 0-th vertex to a j-th point
    lpt = vector_lengths(pt)
    # The distance of j-th point from the segment, pt*sin(phi)
    psinph = (seg[1]*pt[0] - seg[0]*pt[1])/lseg
    #print psinph
    if psinph > dmaxr:
        dmaxr = psinph
        jmaxr = j
    if psinph < dmaxl:
        dmaxl = psinph
        jmaxl = j

print 'jmaxr = ', jmaxr, ', dmaxr = ', dmaxr, ', r[jmaxr,:] = ', r[jmaxr,:]
print 'jmaxl = ', jmaxl, ', dmaxl = ', dmaxl, ', r[jmaxl,:] = ', r[jmaxl,:]

#
# Add the leftmost and rightmost points to the border
#
pbd = 1
if dmaxr >= 0:
    bd.insert(pbd,jmaxr)  # Insert the rightmost point before pbd = 1
    ix[jmaxr] = 2  # Remove the point from pool and add to border
    pbd = pbd + 1
if dmaxl <= 0:
    pbd = pbd + 1
    bd.insert(pbd,jmaxl)  # Insert the rightmost point before pbd = 1
    ix[jmaxl] = 2  # Remove the point from pool and add to border

print 'ix =', ix
print 'bd = ', bd
nbd = len(bd)
#
# Plot the border
#
plot_hull(bd, ix, r)

#show(); sys.exit(0)
    
#
# Traveling CCW along the border, mark all the points: 
# x[exterior] = 1
# x[interior] = 0
# x[border] = 2
#
print 'ix = ', ix
#show(); sys.exit(0)


nbd = len(bd)
bd1 = copy.copy(bd)
extr = False # Assume no exterior points
pbd = 0 # Pointer to the end of segment

# Seed all the non-zero and non-2 points with -1
for i in xrange(N):
    if (ix[i] != 0) and (ix[i] != 2): ix[i] = -1
    
for i in xrange(nbd):
    i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
    pbd = pbd + 1
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
        #print psinph
        if psinph >= 0.0:
            ix[j] = 1 # Mark r[j,:] as exterior point
            print 'Mark r[j,:] as exterior point: ix[j] = 1, j = ', j
            if psinph > dmaxr:  # Search for the ext. point farthest from seg.
                dmaxr = psinph
                jmaxr = j
                print 'jmaxr = ', j
    if dmaxr >= 0: # Farthest exterior point found
        print 'inserted jmaxr = ', jmaxr, ', between ', bd[i], ' and ', bd[i1]
        bd1.insert(pbd,jmaxr) # Insert a new vortex between i-th and (i+1)-th
        pbd = pbd + 1
        ix[jmaxr] = 2  # Mark as border point

#
# Plot new border
#
print 'New border:'
nbd = len(bd1)
plot_hull(bd1, ix, r)

#show(); sys.exit(0)
#
# At this point all the exterior points are marked ix = 1.
# The border is marked as ix = 2. All other points belong to the interior,
# and they have retained ix = -1.
# Mark all the interior (i.e. -1) points with ix = 0 to avoid their processing.
# Also, if all the non-border points are alteady inside the polygon,
# the extr flag will remain False -- and we are done.
#
extr = False
for j in xrange(N):
    if ix[j] == -1: ix[j] = 0
    if ix[j] ==  1:
        extr = True
        #ix[j] = -1 # The points outside the initial tetragon

if extr: print 'The exterior points still exist'
print 'bd1 = ', bd1
print 'ix = ', ix

plot_hull(bd1, ix, r)

#show(); sys.exit(0)

#
# Main loop
#
#bd0 = copy.copy(bd)
#for ii in (1,):
while extr:
    bd = bd1
    bd1 = copy.copy(bd)
    nbd = len(bd)
    print 'nbd = ', nbd
    pbd = 0 # Pointer to the end of segment
    # Seed all the non-zero and non-2 points with -1
    for i in xrange(N):
        if (ix[i] != 0) and (ix[i] != 2): ix[i] = -1
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
            print 'j = ', j, ', psinph = ', psinph, ', dmaxr = ', dmaxr
            if psinph >= 0.0:
                ix[j] = 1 # Mark r[j,:] as exterior point
                if psinph > dmaxr:  # Ext. point farthest from seg.
                    dmaxr = psinph
                    jmaxr = j
        if dmaxr >= 0: # Farthest exterior point found
            print 'inserted jmaxr = ', jmaxr, ', between ', bd[i], \
                  ' and ', bd[i1]
            bd1.insert(pbd,jmaxr) # Insert new vort. bw i-th and (i+1)-th
            pbd = pbd + 1
            ix[jmaxr] = 2  # Mark as border point
    extr = False
    for j in xrange(N):
        if ix[j] == -1: ix[j] = 0
        if ix[j] ==  1:
            extr = True
            #ix[j] = -1 # The points outside the current pentagon


bd = bd1
#
# Plot new border
#
#
# Plot new border
#
print 'New border:'
nbd = len(bd1)
plot_hull(bd1, ix, r)
#print 'bd0 = ', bd0
print 'bd1 = ', bd1
print 'ix = ', ix










show()
