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

figure(figsize=(18,6))
plot([-Nh2,Nh2], [0,0], 'k')  # X axis
plot([0,0], [-Nh2,Nh2], 'k')  # Y axis
#axis('equal')

# Ticks
for i in xrange(Nh+1): 
    plot([i+1,i+1], [-0.1,0.1], 'k')
    plot([-i-1,-i-1], [-0.1,0.1], 'k')
    plot([-0.02,0.02], [i+1,i+1], 'k')
    plot([-0.02,0.02], [-i-1,-i-1], 'k')

# Tick labels
text(0.85, -0.85, r'$v$')
text(-1.15, -0.85, r'$v$')
for i in xrange(Nh):  
    text(i+1.9, -0.85, r'$%2dv$' % (i+2), fontsize=12)
    text(-i-2.3, -0.85, r'$%2dv$' % (-i-2), fontsize=12)
    text(-0.55, i+0.8, r'$%2d$' % (i+1), fontsize=12)
for i in xrange(1,Nh): # Negative Y tick labels
    text(-0.55, -i-1.2, r'$%2d$' % (i+1), fontsize=12)
## # Tick labels
## text(0.9, -0.5, r'$v$')
## text(-1.1, -0.5, r'$v$')
## for i in xrange(Nh):  
##     text(i+1.9, -0.5, r'$%2dv$' % (i+2), fontsize=12)
##     text(-i-2.3, -0.5, r'$%2dv$' % (-i-2), fontsize=12)
    

for i in xrange(Nh-1):  # Steps
    plot([i,i+1], [i+1,i+1], 'k', lw=lwidth)
    plot([-i,-i-1], [-i-1,-i-1], 'k', lw=lwidth)
    plot([i+1,i+1], [i+1,i+2], 'k', lw=lwidth)
    plot([-i-1,-i-1], [-i-1,-i-2], 'k', lw=lwidth)
    
plot([0,0], [-1,1], 'k', lw=lwidth)
plot([Nh-1,Nh+2], [Nh,Nh], 'k', lw=lwidth)
plot([-Nh+1,-Nh-2], [-Nh,-Nh], 'k', lw=lwidth)



#====================================================================


#
# Delta-functions
#
figure(figsize=(18,6))
plot([-Nh2,Nh2], [0,0], 'k')  # X axis
plot([0,0], [-1.,2.2], 'k')  # Y axis
#axis('equal')

# Ticks
for i in xrange(Nh+1): 
    plot([i+1,i+1], [-0.05,0.05], 'k')
    plot([-i-1,-i-1], [-0.05,0.05], 'k')
    
plot([-0.1,0.1], [2,2], 'k')
plot([-0.1,0.1], [1,1], 'k')
plot([-0.1,0.1], [-1,-1], 'k')

# Tick labels
text(0.9, -0.25, r'$v$')
text(-1.1, -0.25, r'$v$')
for i in xrange(Nh):  
    text(i+1.9, -0.25, r'$%2dv$' % (i+2), fontsize=12)
    text(-i-2.3, -0.25, r'$%2dv$' % (-i-2), fontsize=12)
    
# Y tick labels
text(-0.35, 1.9, r'$2$', fontsize=12)
text(-0.35, 0.9, r'$1$', fontsize=12)
text(-0.6, -1.05, r'$-1$', fontsize=12)
    

plot([0,0], [0,2.], 'k', lw=lwidth)
for i in xrange(Nh-1):  # Delta-functions
    plot([i+1,i+1], [0,1], 'k', lw=lwidth)
    plot([-i-1,-i-1], [0,1], 'k', lw=lwidth)

ylim(-3,3)

# Annotate
text(-1.5,1.2, r'$\delta(x+v)$', fontsize=12)
text(-0.2,2.4, r'$2\delta(x)$', fontsize=12)
text(0.5,1.2, r'$\delta(x-v)$', fontsize=12)
for i in xrange(2,Nh):
    text(-i-0.5,1.2, r'$\delta(x+%dv)$' % i, fontsize=12)
    text(i-0.5,1.2, r'$\delta(x-%dv)$' % i, fontsize=12)



## #====================================================================


## #
## # erfs
## #

## figure(figsize=(18,6))
## plot([-Nh2,Nh2], [0,0], 'k')  # X axis
## plot([0,0], [-1.,2.2], 'k')  # Y axis
## #axis('equal')

## # Ticks
## for i in xrange(Nh+1): 
##     plot([i+1,i+1], [-0.05,0.05], 'k')
##     plot([-i-1,-i-1], [-0.05,0.05], 'k')
    
## plot([-0.1,0.1], [2,2], 'k')
## plot([-0.1,0.1], [1,1], 'k')
## plot([-0.1,0.1], [-1,-1], 'k')

## # Tick labels
## text(0.9, -0.25, r'$v$')
## text(-1.1, -0.25, r'$v$')
## for i in xrange(Nh):  
##     text(i+1.9, -0.25, r'$%2dv$' % (i+2), fontsize=12)
##     text(-i-2.3, -0.25, r'$%2dv$' % (-i-2), fontsize=12)
    
## # Y tick labels
## text(-0.35, 1.9, r'$2$', fontsize=12)
## text(-0.35, 0.9, r'$1$', fontsize=12)
## text(-0.6, -1.05, r'$-1$', fontsize=12)
    

## plot([0,0], [0,2.], 'k', lw=lwidth)
## for i in xrange(Nh-1):  # Delta-functions
##     plot([i+1,i+1], [0,1], 'k', lw=lwidth)
##     plot([-i-1,-i-1], [0,1], 'k', lw=lwidth)

## ylim(-3,3)

## # Annotate
## text(-1.5,1.2, r'$\delta(x+v)$', fontsize=12)
## text(-0.2,2.4, r'$2\delta(x)$', fontsize=12)
## text(0.5,1.2, r'$\delta(x-v)$', fontsize=12)
## for i in xrange(2,Nh):
##     text(-i-0.5,1.2, r'$\delta(x+%dv)$' % i, fontsize=12)
##     text(i-0.5,1.2, r'$\delta(x-%dv)$' % i, fontsize=12)












show()






