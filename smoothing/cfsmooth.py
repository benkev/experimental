#
# Smoothing constant and function juncture
#
import pylab
from pylab import *
import matplotlib.pyplot
from matplotlib.pyplot import *

xl = -3.; xr = 5.
x = linspace(xl, xr, 1001)
y = -(x - 1.)**2 + 8.

eps = 4.
f = zeros_like(y)
g = zeros_like(y)
const = 1.

for i in xrange(1001):
    dy = y[i] - const
    if dy < -eps: f[i] = const
    if dy > +eps: f[i] = y[i]
    if abs(dy) <= eps:
        #g[i] = 2**(-64*(dy/eps)**2 - 1.)
        #g[i] = 1./(1. + 64*(dy/eps)**2)
        ## if dy <= 0.:
        ##     f[i] = g[i] + const
        ## else:
        ##     f[i] = g[i] + y[i]
        f[i] = 0.5*(const*(1. - tanh(2*dy)) + y[i]*(tanh(2*dy) + 1.))




figure()
plot(x, y); grid(1)
plot(x, f); grid(1)
xleft, xright = xlim()
ybot, ytop = ylim()
plot([0, 0],[ybot,ytop], 'k', lw=0.5)
plot([xleft-2.,xright+2.], [0, 0], 'k', lw=0.5)
plot([xleft-2.,xright+2.], [const,const], '-.r')
xlim(-4, 6)

show()

