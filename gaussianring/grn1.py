from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import*
import matplotlib.pyplot as plt

s = .1
cx = 1.
cy = .2

gsn = lambda X,Y,s,cx,cy:(1./(2.*pi*s))*exp(-0.5*((cx*X)**2 + (cy*Y)**2)/s**2)
hump = lambda X,Y,s,cx,cy: 1./((cx*X)**2 + (cy*Y)**2 + 1.)

rx = linspace(-3., 3., 100)
ry = linspace(-3., 3., 100)
X, Y = np.meshgrid(rx,ry)

z = gsn(X, Y, s, cx, cy)
#z = hump(X, Y, s, cx, cy)
#figure(); imshow(z)
## fig = figure()
## ax = fig.add_subplot(111, projection='3d')
## ax.plot_wireframe(X, Y, z, rstride=2, cstride=2, lw=0.4);

## ExpX = 3.*exp(X-3)
## Xi = ExpX*cos(Y)
## Et = ExpX*sin(Y)

#Xi = 0.5*log(X**2 + Y**2)
Xi = log(abs(X))
Et = arctan2(Y,X)

z1 = gsn(Xi, Et, s, cx, cy)
#z1 = hump(Xi, Et, s, cx, cy)

fig = figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, z1, rstride=2, cstride=2, lw=0.4);
#ax.plot_wireframe(Xi, Et, z1, rstride=2, cstride=2, lw=0.4);

figure(figsize=(12,6));
subplot(121); imshow(z, cmap=cm.hot); colorbar(shrink=0.7);
subplot(122); imshow(z1, cmap=cm.hot); colorbar(shrink=0.7);


show()










