from pylab import *


nTracedPts = 201

TracedPts_I = int_(10000*(1.5 + 0.5*randn(nTracedPts)))
TracedPts_I.sort()

print TracedPts_I

#iRay = TracedPts_I[3*nTracedPts/4]
#iRay = TracedPts_I[0]
##iRay = TracedPts_I[-1]
iRay = TracedPts_I[nTracedPts/5]


print 'iRay=', iRay

i0 = 0
i1 = nTracedPts-1

while (i0 < i1):
    im = (i0 + i1)/2
    print 'T[%d]=%d   T[%d]=%d   T[%d]=%d' % (i0, TracedPts_I[i0], \
                                              im, TracedPts_I[im],
                                              i1, TracedPts_I[i1],)
    if TracedPts_I[im] < iRay:
        i0 = im + 1
    else:
        i1 = im

if ((i0 == i1) and (TracedPts_I[i0] == iRay)):
    print 'Found at', i0, ', TracedPts_I[%d] = %d' % (i0, TracedPts_I[i0])
else:
    print 'Not found'

print 'i0=%d, i1=%d' % (i0, i1)
                   
