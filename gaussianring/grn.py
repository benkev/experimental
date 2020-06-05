from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import*
import matplotlib.pyplot as plt

sx = 0.13
sy = 1.2
r = 3.

gsn = lambda X,Y,sx,sy:(1./(2.*pi*sx*sy))*exp(-0.5*((X/sx)**2 + (Y/sy)**2))
gsn1 = lambda X,Y,sx,sy:1. - 0.5*(((X-1.)/sx)**2 + ((Y-1.)/sx)**2)
hump = lambda X,Y,s,cx,cy: 1./((cx*X)**2 + (cy*Y)**2 + 1.)

rng = 2.*pi
rx = linspace(-rng, rng, 200)
ry = linspace(-rng, rng, 200)
X, Y = np.meshgrid(rx,ry)

z = gsn(X, Y, sx, sy)
#z = hump(X, Y, s, cx, cy)
#figure(); imshow(z)
## fig = figure()
## ax = fig.add_subplot(111, projection='3d')
## ax.plot_wireframe(X, Y, z, rstride=2, cstride=2, lw=0.4);

## ExpX = 3.*exp(X-3)
## Xi = ExpX*cos(Y)
## Et = ExpX*sin(Y)

Xi = 0.5*log((X**2 + Y**2)/(r**2))
Et = arctan2(Y,X)


z1 = gsn(Xi, Et, sx, sy)
#z1 = gsn1(X, Y, sx, sy)
#z1 = hump(Xi, Et, s, cx, cy)

#fig = figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(X, Y, z1, rstride=2, cstride=2, lw=0.4);
#ax.plot_wireframe(Xi, Et, z1, rstride=2, cstride=2, lw=0.4);

figure(figsize=(12,6));
subplot(121); imshow(z, cmap=cm.hot); colorbar(shrink=0.7);
subplot(122); imshow(z1, cmap=cm.hot); colorbar(shrink=0.7);


show()










