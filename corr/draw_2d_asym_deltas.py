from pylab import *

lwidth = 2.5

Nb = 4           # N bits
Nl = 2**Nb       # N levels
Nh = Nl/2
N = Nl - 1       # N levels - 2 plus center
#Nh = N/2
Nh2 = Nh + 2

print 'Number of bits: ', Nb
print 'Number of levels: ', Nl
print 'Number of levels actual (minus 1): ', N
print 'Half number of levels ', Nh
print

sx = 8
sy = 4
rho = 0.6
ndis2d = lambda x, y: (1./(2.*pi*sx*sy*sqrt(1-rho**2)))* \
            exp(-(0.5/(1-rho**2))*((x/sx)**2 + (y/sy)**2 - 2*rho*x*y/(sx*sy)))

figure(figsize=(16,16))
plot([-Nh2,Nh2], [0,0], 'k')  # X axis
plot([0,0], [-Nh2,Nh2], 'k')  # Y axis
axis('equal')

# Ticks
for i in xrange(Nh+1): 
    plot([i+1,i+1], [-0.1,0.1], 'k')
    plot([-i-1,-i-1], [-0.1,0.1], 'k')
    plot([-0.1,0.1], [i+1,i+1], 'k')
    plot([-0.1,0.1], [-i-1,-i-1], 'k')

# Tick labels
text(0.85, -0.85, r'$v$')
text(-1.20, -0.85, r'$-v$')
text(-0.55, 0.85, r'$v$')
text(-0.85, -1.1, r'$-v$')
for i in xrange(Nh):  
    text(i+1.9, -0.85, r'$%2dv$' % (i+2), fontsize=12)
    text(-i-2.3, -0.85, r'$%2dv$' % (-i-2), fontsize=12)
for i in xrange(Nh): 
    text(-0.55, i+1.9, r'$%2dv$' % (i+2), fontsize=12)
    text(-0.80, -i-2.1, r'$-%2dv$' % (i+2), fontsize=12)

for i in xrange(-Nh+1,Nh):
    for j in xrange(-Nh+1,Nh):
        plot(i, j, 'ro')

rng = linspace(-Nh2, Nh2, 101)
X, Y = meshgrid(rng, rng)
Z = ndis2d(X,Y)
contour(X, Y, Z, 10, colors='k')

show()
