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

# Normal curve
s = 3.0  # Sigma, STD
ndist = lambda x: (1./(s*sqrt(2.*pi)))*exp(-0.5*(x/s)**2)


#
# erfs
#
size_xt = 0.001
size_yt = 0.05

figure(figsize=(18,10))
plot([-Nh2,Nh2], [0,0], 'k')  # X axis
plot([0,0], [-0.01,0.15], 'k')  # Y axis
#axis('equal')

# Ticks
for i in xrange(Nh+1): 
    plot([i+1,i+1], [-size_xt,size_xt], 'k')
    plot([-i-1,-i-1], [-size_xt,size_xt], 'k')

# Tick labels
text(0.95, -0.006, r'$v$', fontsize=16)
text(-1.25, -0.006, r'$-v$', fontsize=16)
for i in xrange(Nh):  
    text(i+1.9, -0.006, r'$%2dv$' % (i+2), fontsize=16)
    text(-i-2.3, -0.006, r'$%2dv$' % (-i-2), fontsize=16)
    
 
ylim(-0.015,0.15)


col = ['r', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet', 'brown']
for i in xrange(0,Nh):
    xr = linspace(i, i+1, 10)
    fill_between(xr, ndist(xr), color=col[i], alpha=0.8)
    xr = linspace(-i,-i-1, 10)
    fill_between(xr, ndist(xr), color=col[i], alpha=0.8)

# Vertical lines
for i in xrange(1,Nh):    
    plot([i,i], [0,ndist(i)], 'w', lw=2)
    plot([-i,-i], [0,ndist(i)], 'w', lw=2)
     
x = linspace(-10., 10., 101)
xv = arange(-Nh,Nh)  # Discrete voltages
y = ndist(x)

plot(x, y, 'orange', lw=3)
plot(xv, ndist(xv), 'ro')



show()






