from pylab import *

Nb = 4           # N bits
Nl = 2**Nb       # N levels
Nh = Nl/2
N = Nl - 1       # N levels - 2 plus center
Nh = N/2

print 'Number of bits: ', Nb
print 'Number of levels: ', Nl
print 'Number of levels actual (minus 1): ', N
print 'Half number of levels ', Nh
print

a = mat([1 for i in xrange(Nh)] + [2] + [1 for i in xrange(Nh)])
b = a.T

K = b*a
print b*a
print

X = zeros((N,N), dtype=int)
Y = zeros((N,N), dtype=int)

for i in xrange(N):
    r = Nh - i
    for j in xrange(N):
        s = Nh - j
        X[i,j] = r*r + s*s
        Y[i,j] = -2*r*s
        print '%2d, %2d |' % (r,s), 
    print

print
       
for i in xrange(N):
    r = Nh - i
    for j in xrange(N):
        s = Nh - j
        rs = r*s
        if   rs <> 0:
            #print '%3dV^2%+3dV^2*rho | ' % (X[i,j], Y[i,j]),
            print '%3d%+3d*rho | ' % (X[i,j], Y[i,j]),
        elif rs == 0:
            #print '    %3dV^2       | ' % X[i,j],
            print '   %3d     | ' % X[i,j],
    print
    
print
print X
print
print Y
print

