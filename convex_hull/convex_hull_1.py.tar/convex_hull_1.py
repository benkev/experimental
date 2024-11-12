#
# Find the minimal conex hull for a set of n points on a plane
#
from pylab import *
import sys
import copy

N = 100
#r = randn(N,2)  # All the points (x,y)
r = array([[ 0.09675173,  0.76215783],
       [ 0.74588475,  0.55192385],
       [ 0.23657207, -1.08612548],
       [-0.08324396,  0.20737612],
       [ 0.11565061,  0.60479385],
       [-0.5203068 ,  1.53653427],
       [ 0.82251745, -0.30993802],
       [-1.06988518,  0.63084131],
       [-0.30194238, -0.27605152],
       [ 0.12797836, -0.04333752],
       [-0.55678163, -0.78383693],
       [-1.61778193, -0.43879655],
       [-0.31296568, -1.03302267],
       [-0.55692048, -0.31507627],
       [ 0.78611188,  1.56707104],
       [ 2.15253056,  1.62043484],
       [ 0.23639005, -0.51121402],
       [ 0.90990056, -0.48031805],
       [ 0.50080918, -0.14876588],
       [-0.94763463,  0.33393933],
       [-2.13074654, -0.67909653],
       [ 0.62061369, -0.06471089],
       [-0.48835927, -0.62455217],
       [ 0.4584685 , -0.00604662],
       [ 0.58668044,  0.24091582],
       [ 2.01611363,  0.6215417 ],
       [-1.12091025,  0.28243506],
       [ 0.49787212,  0.00996293],
       [ 1.00895536,  0.74289856],
       [-0.39950732, -0.57183192],
       [-0.3410592 ,  1.14404559],
       [ 1.09957376,  0.20482193],
       [-0.56769219, -0.92424613],
       [ 2.74660501, -1.86256657],
       [-0.53868457,  0.16230253],
       [ 0.84722532, -0.7565707 ],
       [-0.81794866, -0.58098566],
       [-0.79662697, -0.88008919],
       [-0.9019009 ,  1.09672115],
       [ 1.30497001, -0.7129829 ],
       [-0.09852203, -1.23467508],
       [-0.99434407, -0.63541168],
       [ 1.5178145 ,  1.43007818],
       [-0.03782845,  0.65472951],
       [ 0.51307737, -0.33733563],
       [ 2.51281053, -0.80090495],
       [ 0.56042406, -0.31263081],
       [ 1.0463647 ,  0.58943716],
       [-0.23998371, -0.06171827],
       [-0.10379708,  0.29003298],
       [-0.20326794, -1.02281028],
       [-0.94597041, -0.50172989],
       [ 1.02486548, -0.54884168],
       [ 1.2191196 ,  1.08052077],
       [-0.40225318, -0.82279413],
       [ 0.70905738, -1.51411981],
       [-1.02995739, -0.21532241],
       [ 0.53435212, -0.49819906],
       [-0.78421808,  0.67140732],
       [ 0.14442246, -0.34453476],
       [-0.45557286,  0.89107653],
       [ 0.97564013,  1.47716114],
       [ 0.21628146, -0.04078375],
       [-0.88872991, -0.50171337],
       [-1.4120076 ,  0.57718109],
       [ 1.22665937,  0.86121251],
       [-0.60128956, -0.45100558],
       [ 2.16464284, -0.76134013],
       [ 0.04958289,  0.62867957],
       [ 1.83324784,  1.80382648],
       [ 1.90180321,  1.87788454],
       [-0.36790871,  1.56667657],
       [-0.05038036, -1.65630449],
       [-1.06882936,  0.59837338],
       [ 2.34374684, -1.14970025],
       [ 0.60493003, -2.90952685],
       [-0.77108783, -0.21730219],
       [ 0.41481689,  0.87823541],
       [ 0.82227787, -1.18570068],
       [ 0.50236035, -1.31782349],
       [-1.35375567,  0.6774195 ],
       [ 0.26151945, -0.37913976],
       [ 0.18503194, -0.0943993 ],
       [ 1.51963342, -0.97998594],
       [-0.6539832 , -0.91147536],
       [-1.23891572,  0.22373325],
       [-1.1981856 ,  1.05983634],
       [ 0.0552567 , -0.07226239],
       [ 0.16850371, -2.23832987],
       [ 0.36365219,  0.03044086],
       [ 1.32009323,  0.07121669],
       [ 0.16973278, -0.28984081],
       [ 1.21346393, -0.81515759],
       [-1.42301676,  0.2358902 ],
       [ 0.00763802, -0.95357815],
       [-1.6939979 ,  1.21022111],
       [-0.1491011 , -1.63500879],
       [ 0.35569573,  0.51261554],
       [-0.01599767, -2.06410822],
       [ 0.36244374,  2.64688697]])
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
#
# Plot the border
#
print 'bd = ', bd
nbd = len(bd)
for i in xrange(nbd):
    i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
    plot((r[bd[i],0], r[bd[i1],0]), (r[bd[i],1], r[bd[i1],1]), 'r')
    print bd[i], r[bd[i],:], '--', bd[i1], r[bd[i1],:]
    

for i in xrange(N):
    if ix[i] == -1: plot(r[i,0], r[i,1], 'b.')
    if ix[i] ==  2: plot(r[i,0], r[i,1], 'ro') # Border points
    if ix[i] ==  0: plot(r[i,0], r[i,1], 'g.')
    if ix[i] ==  1: plot(r[i,0], r[i,1], 'm.')

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

for i in xrange(nbd):
    i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
    pbd = pbd + 1
    seg = r[bd[i1],:] - r[bd[i],:]        # Segment vector
    lseg = vector_lengths(seg) # Its length as sqrt(sum(seg**2))
    dmaxr = -1.0  # Right distance can only be positive
    for j in xrange(N):
        if mod(ix[j],2) == 0: continue # Avoid border (ix=2) and interior (x=0) 
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
for i in xrange(nbd):
    i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
    plot((r[bd1[i],0], r[bd1[i1],0]), (r[bd1[i],1], r[bd1[i1],1]), 'c')
    print bd1[i], r[bd1[i],:], '--', bd1[i1], r[bd1[i1],:]
print 'bd1 = ', bd1
print 'ix = ', ix

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
        ix[j] = -1 # The points outside the initial tetragon

if extr: print 'The exterior points still exist'
print 'bd1 = ', bd1
print 'ix = ', ix

#show(); sys.exit(0)

#
# Main loop
#
while extr:
    bd = bd1
    bd1 = copy.copy(bd)
    nbd = len(bd)
    print 'nbd = ', nbd
    pbd = 0 # Pointer to the end of segment
    for i in xrange(nbd):
        pbd = pbd + 1
        i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
        seg = r[bd[i1],:] - r[bd[i],:]        # Segment vector
        lseg = vector_lengths(seg) # Its length as sqrt(sum(seg**2))
        dmaxr = -1.0  # Right distance can only be positive
        for j in xrange(N):
            if mod(ix[j],2) == 0: continue # Avoid border and interior (x=0,2) 
            pt = r[j,:] - r[bd[i],:] # Vector from 0-th vertex to a j-th point
            lpt = vector_lengths(pt)
            # Left(-) or right(+) position of j-th point wrt to i-th segment
            psinph = (seg[1]*pt[0] - seg[0]*pt[1])/lseg  # lpt*sin(phi)
            print psinph
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
    #show(); sys.exit(0)
    extr = False
    for j in xrange(N):
        if ix[j] == -1: ix[j] = 0
        if ix[j] ==  1:
            extr = True
            ix[j] = -1 # The points outside the current pentagon

for i in xrange(N):
    if ix[i] == -1: plot(r[i,0], r[i,1], 'b.')
    if ix[i] ==  2: plot(r[i,0], r[i,1], 'ro') # Border points
    if ix[i] ==  0: plot(r[i,0], r[i,1], 'g.')
    if ix[i] ==  1: plot(r[i,0], r[i,1], 'm.')

bd = bd1
#
# Plot new border
#
print 'New border:'
nbd = len(bd1)
for i in xrange(nbd):
    i1 = mod((i+1), nbd)   # If i+1 < nbd, i1 = i+1, if i == nbd, i1 = 0
    plot((r[bd1[i],0], r[bd1[i1],0]), (r[bd1[i],1], r[bd1[i1],1]), 'c')
    print bd1[i], r[bd1[i],:], '--', bd1[i1], r[bd1[i1],:]
print 'bd1 = ', bd1
print 'ix = ', ix










show()
